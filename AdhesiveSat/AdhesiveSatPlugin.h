#ifndef ADHESIVESATPLUGIN_H
#define ADHESIVESATPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "AdhesiveSatDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {

    class Simulator;
    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class ParallelUtilsOpenMP;

    template <class T> 
    class Field3D;

    template <class T> 
    class WatchableField3D;

    class ADHESIVESAT_EXPORT AdhesiveSatData {
    public:
        int: occupiedAdhesiveSites;

        AdhesiveSatData() :
            occupiedAdhesiveSites(0) {
            }
    }

    class ADHESIVESAT_EXPORT  AdhesiveSatPlugin : public Plugin ,public EnergyFunction ,public CellGChangeWatcher {

    private:            
        CC3DXMLElement *xmlData;        
        Potts3D *potts;
        Simulator *sim;
        ParallelUtilsOpenMP *pUtils;            
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        
        Automaton *automaton;
        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;

        void initializeYField();
        void initializeOccuppiedSiteCounts();
        bool isAdhesiveSite(const Point3D& pt) const;
        int getOccupiedSiteCount(const CellG* cell);
        double adhesionEnergy(double occupiedArea) const;

        Field3D<unsigned char>* adhesionField;
        std::string adhesionFieldName;
		ExtraMembersGroupAccessor<AdhesiveSatData>
            adhesiveSatDataAccessor;

		double E0;
        double Aref;

    public:
        AdhesiveSatPlugin();
        virtual ~AdhesiveSatPlugin();
        
        virtual void init(
            Simulator *simulator, 
            CC3DXMLElement *_xmlData=0
        );

        virtual void extraInit(Simulator *simulator);


        //Energy function interface
        virtual double changeEnergy(
            const Point3D &pt, 
            const CellG *newCell, 
            const CellG *oldCell
        );        
        // CellChangeWatcher interface
        virtual void field3DChange(
            const Point3D &pt, 
            CellG *newCell, 
            CellG *oldCell
        );
                

        

        //Steerrable interface

        virtual void update(
            CC3DXMLElement *_xmlData, 
            bool _fullInitFlag=false
        );
        virtual std::string steerableName();

        virtual std::string toString();

        ExtraMembersGroupAccessor<AdhesiveSatData>*
            getAdhesiveSatDataAccessorPtr() {
            return &adhesiveSatDataAccessor;
        }

    };

};

#endif
