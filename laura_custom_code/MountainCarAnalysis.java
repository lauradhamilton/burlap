import burlap.behavior.singleagent.learning.GoalBasedRF;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.visualizer.Visualizer;

//Map continuous space to a grid
import burlap.behavior.singleagent.auxiliary.StateGridder;

public class MountainCarAnalysis {

    MountainCar			mcGen;
    Domain			domain;
    StateParser			sp;
    RewardFunction		rf;
    TerminalFunction		tf;
    StateConditionTest		goalCondition;
    State			initialState;
    DiscreteStateHashFactory    hashingFactory;

    public MountainCarAnalysis() {
        
        MountainCar mcGen = new MountainCar();
        Domain domain = mcGen.generateDomain();
        TerminalFunction tf = mcGen.new ClassicMCTF();
        RewardFunction rf = new GoalBasedRF(tf, 100);

        StateGenerator rStateGen = new MCRandomStateGenerator(domain);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
        SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20, tf, null);

        ConcatenatedObjectFeatureVectorGenerator fvGen = new ConcatenatedObjectFeatureVectorGenerator(true, MountainCar.CLASSAGENT);
        FourierBasis fb = new FourierBasis(fvGen, 4);

        LSPI lspi = new LSPI(domain, rf, tf, 0.99, fb);
        lspi.setDataset(dataset);

        lspi.runPolicyIteration(30, 1e-6);

        GreedyQPolicy p = new GreedyQPolicy(lspi);

        gridder.setObjectClassAttributesToTile(MountainCar.CLASSAGENT,
            new AttributeSpecification(domain.getAttribute(MountainCar.ATTX), 4),
            new AttributeSpecification(domain.getAttribute(MountainCar.ATTV), 3));

        Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
        VisualActionObserver vaob = new VisualActionObserver(domain, v);
        vaob.initGUI();
        ((SADomain)domain).addActionObserverForAllAction(vaob);

        State s = mcGen.getCleanState(domain);
        while(true){
            p.evaluateBehavior(s, rf, tf);
        }

    }
}

