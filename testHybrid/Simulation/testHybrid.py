
from cc3d import CompuCellSetup
        

from testHybridSteppables import testHybridSteppable, FiberPlotterSteppable

# CompuCellSetup.register_steppable(steppable=testHybridSteppable(frequency=1))
hybrid_steppable = testHybridSteppable(frequency=1)
CompuCellSetup.register_steppable(steppable=hybrid_steppable)

plotter_steppable = FiberPlotterSteppable(hybrid_steppable=hybrid_steppable,frequency=10)
CompuCellSetup.register_steppable(steppable=plotter_steppable)


CompuCellSetup.run()
