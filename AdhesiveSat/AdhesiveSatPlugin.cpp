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
    lockPtr(0),
// xmlData(0) ,
// boundaryStrategy(0)
{}

AdhesiveSatPlugin::~AdhesiveSatPlugin() {

    if (pUtils && lockPtr) {
        pUtils->destroyLock(lockPtr);
        delete lockPtr;
        lockPtr = nullptr;
    }


    delete lockPtr;

    lockPtr=0;

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
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = static cast<WatchableField3D<CellG*> *>(
        potts->getCellFieldG()
    );
    
    fieldDim = cellFieldG->getDim();

    ASSERT_OR_THROW(
        "AdhesiveSat requires a 200x200x1 lattice",
        fieldDim.x == 200 &&
        fieldDim.y == 200 &&
        fieldDim.z == 1

    )


    potts->getCellFactoryGroupPtr()->registerClass(
        &adhesiveSatDataAccessor
    );

    /*
    Create numpy backed scalar field
    unsigned char used as binary type
    0 = False
    1 = True
    */
    sim->createGenericScalarField<unsigned char>(
        adhesionFieldName
    );

    adhesionField=
        sim->getGenericScalarField<unsigned char>(
            adhesionFieldName
        );

    xmlData=_xmlData;
    


    pUtils=sim->getParallelUtils();

    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;

    pUtils->initLock(lockPtr); 

    update(_xmlData, true);

    potts->registerEnergyFunctionWithName(this,"AdhesiveSat");

    // field3dChange called after successful pixel changes
    potts->registerCellGChangeWatcher(this);    
    simulator->registerSteerableObject(this);

}

            

void AdhesiveSatPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 

{

    //This function will be called after each succesful pixel copy - field3DChange does usuall ohusekeeping tasks to make sure state of cells, and state of the lattice is uptdate
    if (!isAdhesiveSite(pt)) { //no point evaluating if not an adhesive site
		return;
	}
    pUtils->setLock(lockPtr); 
	if (oldCell) {
		AdhesiveSatData* oldData =adhesiveSatDataAccessor.get(
                	oldCell->extraAttribPtr);
		--oldData->occupiedAdhesiveSites; //old cell loses a site
    	}
	if (newCell) {
        		AdhesiveSatData* newData =adhesiveSatDataAccessor.get(
               	newCell->extraAttribPtr);
		++newData->occupiedAdhesiveSites; //new cell gains a site
    	}
    pUtils->unsetLock(lockPtr); 
    

}


void AdhesiveSatPlugin::initializeYField(){
    const int centerX = 100;
    const int halfWidth = 3;
    
    for (int x = 0; x <fieldDim.x; ++x){
        for (int y = 0; y <fieldDim.y; ++y){
            Point3D pt(x,y,0)
            const bool stem = 
                y >= 20 &&
                y >= 100 &&
                std::abs(x - centerX) <= halfWidth;

            const bool leftArm =
                y >= 100 &&
                y <= 160 &&
                std::abs((x + y) - 200) <= halfWidth;
            
            const bool rightArm = 
                y >= 100 &&
                y <= 160 &&
                std::abs(x - y) <= halfWidth;

            const bool adhesionMoleculesPresent =
                stem || leftArm || rightArm;

            adhesionField->set(
                pt,
                static_cast<unsigned char>(
                    adhesionMoleculesPresent
                )
            );
        }
    }
}


bool AdhesiveSatPlugin::isAdhesiveSite(const Point3D& pt) const {
    if (!adhesionField->isValid(pt)) {
        return False;
    }
    return adhesionField->get(pt) != 0;
}


void AdhesiveSatPlugin::initializeOccuppiedSiteCounts(){
    CellInventory* CellInventoryPtr = 
        &potts->getCellInventory();

    CellInventory::cellInventoryIterator cInvItr;

    for (
        cInvItr = CellInventoryPtr->cellInventoryBegin();
        cInvItr != CellInventoryPtr->cellInventoryEnd();
        ++cInvItr
    ) {
        CellG* cell = 
            CellInventoryPtr->getCell(cInvItr);

        AdhesiveSatData* data = 
            adhesiveSatDataAccessor.get(
                cell->extraAttribPtr
            );
        data->occupiedAdhesiveSites = 0;
    }

    for (int x = 0; x <fieldDim.x; ++x){
        for (int y = 0; y <fieldDim.y; ++y){
            Point3D pt(x,y,0)
            if (!isAdhesiveSite(pt)) {
                continue;
            }
            CellG* cell = cellFieldG->get(pt);

            if (!cell){
                continue
            }

            AdhesiveSiteData* data = 
                adhesiveSatDataAccessor.get(
                    cell->extraAttribPtr
                );
            ++data->occupiedAdhesiveSites;
        }
    }
}



double AdhesiveSatPlugin::adhesionEnergy(double occupiedArea) const {
    if (occupiedArea <= 0.0){
        return 0.0;
    }
	// return -E0 * occupiedArea /(Aref + occupiedArea);
    return -E0 * occupiedArea /(Aref + occupiedArea);
}

double AdhesiveSatPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	if (!isAdhesiveSite(pt)) { //no point evaluating if not an adhesive site
        		return 0.0;
    	}
    double deltaH = 0.0; //initialize energy
    if (oldCell) {
            const int oldAi =getOccupiedSiteCount(oldCell);
            const int proposedAi = oldAi - 1; //old cell would lose a site
            deltaH += adhesionEnergy(proposedAi) - adhesionEnergy(oldAi);
        }
    if (newCell) {
        const int oldAi =getOccupiedSiteCount(newCell); 
        const int proposedAi = oldAi + 1; //new cell would gain a site
            deltaH +=adhesionEnergy(proposedAi) -adhesionEnergy(oldAi);
        }
        return deltaH;
}



void AdhesiveSatPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    if (!_xmlData){
        return;
    }
    CC3DXMLElement* e0Element = _xmlData->getFirstElement("E0");

    if (e0Element) {
        E0 = e0Element->getDouble();
    }

    CC3DXMLElement* aRefElement = _xmlData->getFirstElement("Aref");

    if (aRef) {
        Aref = aRefElement->getDouble();
    }

    ASSERT_OR_THROW(
        "AdhesiveStat Aref must be greater than zero"
        Aref > 0
    );


//     //PARSE XML IN THIS FUNCTION

//     //For more information on XML parser function please see CC3D code or lookup XML utils API

//     automaton = potts->getAutomaton();

//     ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

//    set<unsigned char> cellTypesSet;


//     if (exampleXMLElem){

//         double param=exampleXMLElem->getDouble();

//         cerr<<"param="<<param<<endl;

//         if(exampleXMLElem->findAttribute("Type")){

//             std::string attrib=exampleXMLElem->getAttribute("Type");

//             // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double

//             cerr<<"attrib="<<attrib<<endl;

//         }

//     }

//     //boundaryStrategy has information aobut pixel neighbors 

//     boundaryStrategy=BoundaryStrategy::getInstance();

}



void AdhesiveSatPlugin::extraInit(Simulator* simulator) {
	initializeYField();
	//this function creates a 200x200 lattice forming a Y with thickness 7
	initializeOccupiedSiteCounts();
	//this function initializes the number of lattice sites already occupied by cells
}

std::string AdhesiveSatPlugin::toString(){
    return "AdhesiveSat";
}

std::string AdhesiveSatPlugin::steerableName(){
    return toString();
}

