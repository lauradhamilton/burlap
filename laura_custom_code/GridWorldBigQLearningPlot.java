import burlap.domain.singleagent.gridworld.*;
import burlap.oomdp.core.*;
import burlap.behavior.singleagent.auxiliary.performance.*;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.behavior.singleagent.learning.*;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.oomdp.singleagent.*;
import burlap.oomdp.singleagent.common.SinglePFTF;

public class GridWorldBigQLearningPlot{

    public static void main(String [] args){

        GridWorldDomain gw = new GridWorldDomain(50,50); 

        //Make some walls
        int[][] map = gw.getMap();
        gw.horizontalWall(0,8,10);
        gw.horizontalWall(10,49,10);
        gw.horizontalWall(0,22,20);
        gw.horizontalWall(24,49,20);
        gw.horizontalWall(0,45,30);
        gw.horizontalWall(47,49,30);
        gw.horizontalWall(0,9,40);
        gw.horizontalWall(11,49,40);

        gw.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate
        final Domain domain = gw.generateDomain(); //generate the grid world domain

        //setup initial state
        State s = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(s, 0, 0);
        GridWorldDomain.setLocation(s, 0, 49, 49);

        //ends when the agent reaches a location
        final TerminalFunction tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));

        //reward function
        final RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), 5., -0.1);

        //initial state generator
        final ConstantStateGenerator sg = new ConstantStateGenerator(s);

        //set up the state hashing system for looking up states
        final DiscreteStateHashFactory hashingFactory = new DiscreteStateHashFactory();

        //Create factory for Q-learning agent
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
            @Override
            public String getAgentName() {
                return "Q-learning";
            }

            @Override
            public LearningAgent generateAgent() {
                return new QLearning(domain, rf, tf, 1, hashingFactory, 0., 0.9);
            }
        };

        //define experiment
        //LearningAlgorithmExperimenter(SADomain domain, RewardFunction rf, StateGenerator sg, int nTrials, int trialLength, LearningAgentFactory... agentFactories)
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter((SADomain)domain, rf, sg, 10, 10000, qLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 10000, TrialMode.MOSTRECENTANDAVERAGE, PerformanceMetric.CUMULATIVESTEPSPEREPISODE, PerformanceMetric.AVERAGEEPISODEREWARD);

        //start experiment
        exp.startExperiment();

    }

}
