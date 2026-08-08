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

/*
Initialize shape
*/
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

// void AdhesiveSatPlugin::initializeGridField() {
//     const int fiberSpacing = 40; // Distance between parallel fiber centers (pixels)
//     const int halfWidth = 1;    // Fiber width = 3 pixels (center pixel +/- halfWidth)

//     for (int x = 0; x < fieldDim.x; ++x) {
//         for (int y = 0; y < fieldDim.y; ++y) {
//             Point3D pt(x, y, 0);

//             int xMod = x % fiberSpacing;
//             int yMod = y % fiberSpacing;

//             // Vertical fibers running along the Y-axis
//             bool verticalFiber = (xMod <= halfWidth) || (xMod >= fiberSpacing - halfWidth);

//             // Horizontal fibers running along the X-axis
//             bool horizontalFiber = (yMod <= halfWidth) || (yMod >= fiberSpacing - halfWidth);

//             bool adhesionMoleculesPresent = verticalFiber || horizontalFiber;

//             adhesionField->set(
//                 pt,
//                 static_cast<unsigned char>(adhesionMoleculesPresent)
//             );
//         }
//     }
// }


/*
Is it an adhesive site
*/
bool AdhesiveSatPlugin::isAdhesiveSite(const Point3D& pt) const {
    if (!adhesionField->isValid(pt)) {
        return false;
    }
    return adhesionField->get(pt) != 0;
}

/*
Occupied site counts
*/
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

/*
Adhesion energy calculation
*/
double AdhesiveSatPlugin::adhesionEnergy(double occupiedArea) const {
    if (occupiedArea <= 0.0) {
        return 0.0;
    }
    return -E0 * occupiedArea / (Aref + occupiedArea); //lambda_C (A / (A + Ah))
}

/*
Focal adhesion penalty
*/
double AdhesiveSatPlugin::calculateFAPenalty(const Point3D& pt) const{
    if (!integrinField || !integrinField->isValid(pt)) {
        return 0.0;
    }

    double N = integrinField->get(pt); // Current integrin count at site pt
    
    // Penalty only applies if integrins exceed baseline N0
    if (N <= N0) {
        return 0.0;
    }

    // Delta H_FA = lambda_FA * (N - N0) / (Nh + N - N0)
    return lambdaFA * (N - N0) / (Nh + N - N0);
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
    // Add the FA penalty only if there is a retraction
    if (oldCell && oldCell != newCell) {
        deltaH += calculateFAPenalty(pt);
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

    // Focal adhesion params
    CC3DXMLElement* lambdaFAElem = _xmlData->getFirstElement("LambdaFA"); //800
    if (lambdaFAElem) lambdaFA = lambdaFAElem->getDouble();

    CC3DXMLElement* n0Elem = _xmlData->getFirstElement("N0");
    if (n0Elem) N0 = n0Elem->getDouble();

    CC3DXMLElement* nhElem = _xmlData->getFirstElement("Nh");
    if (nhElem) Nh = nhElem->getDouble();
}

void AdhesiveSatPlugin::initializeBeads(double stepSize) {
    beadPositions.clear();
    const double centerX = 100.0;

    // Stem: x = 100, y from 20 to 100
    for (double y = 20.0; y <= 100.0; y += stepSize) {
        beadPositions.push_back({centerX, y, 0.0});
    }

    // Left arm: x = 200 - y, y from 100 + step to 160
    for (double y = 100.0 + stepSize; y <= 160.0; y += stepSize) {
        double x = 200.0 - y;
        beadPositions.push_back({x, y, 0.0});
    }

    // Right arm: x = y, y from 100 + step to 160
    for (double y = 100.0 + stepSize; y <= 160.0; y += stepSize) {
        double x = y;
        beadPositions.push_back({x, y, 0.0});
    }
}

void AdhesiveSatPlugin::extraInit(Simulator* simulator) {
    initializeYField();
    // initializeGridField();
    initializeBeads(1.0); // Creates beads every 1.0 unit along fibers
    initializeOccupiedSiteCounts();
}

std::string AdhesiveSatPlugin::toString() {
    return "AdhesiveSat";
}

std::string AdhesiveSatPlugin::steerableName() {
    return toString();
}

