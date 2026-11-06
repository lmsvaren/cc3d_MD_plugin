
from cc3d import CompuCellSetup
        

from hybridCPMSteppables import hybridCPMSteppable, FiberPlotterSteppable

# CompuCellSetup.register_steppable(steppable=hybridCPMSteppable(frequency=1))

hybrid_steppable = hybridCPMSteppable(frequency=1)
CompuCellSetup.register_steppable(steppable=hybrid_steppable)

plotter_steppable = FiberPlotterSteppable(hybrid_steppable=hybrid_steppable,frequency=10)
CompuCellSetup.register_steppable(steppable=plotter_steppable)


CompuCellSetup.run()
