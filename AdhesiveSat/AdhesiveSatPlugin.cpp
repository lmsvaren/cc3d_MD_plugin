#include <CompuCell3D/CC3D.h>        
#include "AdhesiveSatPlugin.h"
#include <algorithm>
#include <cmath>

using namespace CompuCell3D;

AdhesiveSatPlugin::AdhesiveSatPlugin():
    sim(nullptr),
    potts(nullptr),
    cellFieldG(nullptr),
    adhesionField(nullptr),
    adhesionFieldName("AdhesionSites"),
    E0(50.0),
    Aref(250.0),
    pUtils(nullptr),
    lockPtr(nullptr),
    xmlData(nullptr),
    automaton(nullptr),
    boundaryStrategy(nullptr)
{}

AdhesiveSatPlugin::~AdhesiveSatPlugin() {
    if (pUtils && lockPtr) {
        pUtils->destroyLock(lockPtr);
        delete lockPtr;
        lockPtr = nullptr;
    }
}

int AdhesiveSatPlugin::getOccupiedSiteCount(const CellG* cell) {
    if (!cell) {
        return 0;
    }
    pUtils->setLock(lockPtr);

    const int result = 
        adhesiveSatDataAccessor.get(
            cell->extraAttribPtr
        )->occupiedAdhesiveSites;

    pUtils->unsetLock(lockPtr);

    return result;
}

void AdhesiveSatPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = static_cast<WatchableField3D<CellG*> *>(
        potts->getCellFieldG()
    );
    
    fieldDim = cellFieldG->getDim();

    ASSERT_OR_THROW(
        "AdhesiveSat requires a 200x200x1 lattice",
        fieldDim.x == 200 &&
        fieldDim.y == 200 &&
        fieldDim.z == 1
    );

    potts->getCellFactoryGroupPtr()->registerClass(
        &adhesiveSatDataAccessor
    );

    sim->createGenericScalarField<unsigned char>(
        adhesionFieldName
    );

    adhesionField = sim->getGenericScalarField<unsigned char>(
        adhesionFieldName
    );

    xmlData = _xmlData;
    pUtils = sim->getParallelUtils();

    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 

    update(_xmlData, true);

    potts->registerEnergyFunctionWithName(this, "AdhesiveSat");

    // field3dChange called after successful pixel changes
    potts->registerCellGChangeWatcher(this);    
    simulator->registerSteerableObject(this);
}

void AdhesiveSatPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
    if (!isAdhesiveSite(pt)) { 
        return;
    }
    pUtils->setLock(lockPtr); 
    if (oldCell) {
        AdhesiveSatData* oldData = adhesiveSatDataAccessor.get(oldCell->extraAttribPtr);
        --oldData->occupiedAdhesiveSites; 
    }
    if (newCell) {
        AdhesiveSatData* newData = adhesiveSatDataAccessor.get(newCell->extraAttribPtr);
        ++newData->occupiedAdhesiveSites; 
    }
    pUtils->unsetLock(lockPtr); 
}

void AdhesiveSatPlugin::initializeYField() {
    const int centerX = 100;
    const int halfWidth = 3;
    
    for (int x = 0; x < fieldDim.x; ++x) {
        for (int y = 0; y < fieldDim.y; ++y) {
            Point3D pt(x, y, 0);
            const bool stem = 
                y >= 20 &&
                y <= 100 &&
                std::abs(x - centerX) <= halfWidth;

            const bool leftArm =
                y >= 100 &&
                y <= 160 &&
                std::abs((x + y) - 200) <= halfWidth;
            
            const bool rightArm = 
                y >= 100 &&
                y <= 160 &&
                std::abs(x - y) <= halfWidth;

            const bool adhesionMoleculesPresent = stem || leftArm || rightArm;

            adhesionField->set(
                pt,
                static_cast<unsigned char>(adhesionMoleculesPresent)
            );
        }
    }
}

bool AdhesiveSatPlugin::isAdhesiveSite(const Point3D& pt) const {
    if (!adhesionField->isValid(pt)) {
        return false;
    }
    return adhesionField->get(pt) != 0;
}

void AdhesiveSatPlugin::initializeOccupiedSiteCounts() {
    CellInventory* cellInventoryPtr = &potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;

    for (
        cInvItr = cellInventoryPtr->cellInventoryBegin();
        cInvItr != cellInventoryPtr->cellInventoryEnd();
        ++cInvItr
    ) {
        CellG* cell = cellInventoryPtr->getCell(cInvItr);

        AdhesiveSatData* data = adhesiveSatDataAccessor.get(cell->extraAttribPtr);
        data->occupiedAdhesiveSites = 0;
    }

    for (int x = 0; x < fieldDim.x; ++x) {
        for (int y = 0; y < fieldDim.y; ++y) {
            Point3D pt(x, y, 0);
            if (!isAdhesiveSite(pt)) {
                continue;
            }
            CellG* cell = cellFieldG->get(pt);

            if (!cell) {
                continue;
            }

            AdhesiveSatData* data = adhesiveSatDataAccessor.get(cell->extraAttribPtr);
            ++data->occupiedAdhesiveSites;
        }
    }
}

double AdhesiveSatPlugin::adhesionEnergy(double occupiedArea) const {
    if (occupiedArea <= 0.0) {
        return 0.0;
    }
    return -E0 * occupiedArea / (Aref + occupiedArea);
}

double AdhesiveSatPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    if (!isAdhesiveSite(pt)) { 
        return 0.0;
    }
    double deltaH = 0.0; 
    if (oldCell) {
        const int oldAi = getOccupiedSiteCount(oldCell);
        const int proposedAi = oldAi - 1; 
        deltaH += adhesionEnergy(proposedAi) - adhesionEnergy(oldAi);
    }
    if (newCell) {
        const int oldAi = getOccupiedSiteCount(newCell); 
        const int proposedAi = oldAi + 1; 
        deltaH += adhesionEnergy(proposedAi) - adhesionEnergy(oldAi);
    }
    return deltaH;
}

void AdhesiveSatPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    if (!_xmlData) {
        return;
    }
    CC3DXMLElement* e0Element = _xmlData->getFirstElement("E0");

    if (e0Element) {
        E0 = e0Element->getDouble();
    }

    CC3DXMLElement* aRefElement = _xmlData->getFirstElement("Aref");

    if (aRefElement) {
        Aref = aRefElement->getDouble();
    }

    ASSERT_OR_THROW(
        "AdhesiveSat Aref must be greater than zero",
        Aref > 0
    );
}

void AdhesiveSatPlugin::extraInit(Simulator* simulator) {
    initializeYField();
    initializeOccupiedSiteCounts();
}

std::string AdhesiveSatPlugin::toString() {
    return "AdhesiveSat";
}

std::string AdhesiveSatPlugin::steerableName() {
    return toString();
}

