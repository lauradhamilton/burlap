//Grid World and core packages
import burlap.behavior.singleagent.*;
import burlap.domain.singleagent.gridworld.*;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.*;
import burlap.oomdp.singleagent.common.*;
import burlap.behavior.statehashing.DiscreteStateHashFactory;

//For visualizations
import burlap.oomdp.visualizer.Visualizer;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;

//Packages for the planning and learning algorithms
import burlap.behavior.singleagent.learning.*;
import burlap.behavior.singleagent.learning.tdmethods.*;
import burlap.behavior.singleagent.planning.*;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.*;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;

//Line visualization
import burlap.oomdp.singleagent.common.VisualActionObserver;

//Value function and policy visualization
import java.awt.Color;
import java.util.List;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.*;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;

//Experimenter tools and performance plotting
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;

//Different exploration strategy for Q-learning
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learnbydemo.apprenticeship.ApprenticeshipLearning.RandomPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyDeterministicQPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.EpsilonGreedy;

public class GridWorldGraphs {

    GridWorldDomain		gwdg;
    Domain			domain;
    StateParser			sp;
    RewardFunction		rf;
    TerminalFunction		tf;
    StateConditionTest		goalCondition;
    State			initialState;
    DiscreteStateHashFactory    hashingFactory;

    public GridWorldGraphs() {
        gwdg = new GridWorldDomain(11, 11); //Must be 11x11 for four rooms layout
        gwdg.setMapToFourRooms(); //four rooms layout
        gwdg.setProbSucceedTransitionDynamics(0.8); //Stochastic transitions have an 0.8 success rate
        domain = gwdg.generateDomain();

        //create the state parser
        sp = new GridWorldStateParser(domain);

        //define the task
        rf = new UniformCostRF();
        tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
        goalCondition = new TFGoalCondition(tf);

        //set up the initial state of the task
        initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(initialState, 0, 0);
        GridWorldDomain.setLocation(initialState, 0, 10, 10);

        //set up the state hashing system
        hashingFactory = new DiscreteStateHashFactory();
        hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT, domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);

        //add visual observer
        VisualActionObserver observer = new VisualActionObserver(domain, GridWorldVisualizer.getVisualizer(gwdg.getMap()));
        ((SADomain)this.domain).setActionObserverForAllAction(observer);
        observer.initGUI();

    }

    public void visualize(String outputPath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, domain, sp, outputPath);
    }

    //Hook up the class constructor and visualizer method to the main class
    public static void main(String[] args) {
        GridWorldGraphs example = new GridWorldGraphs();
        String outputPath = "../../../results/grid-world"; //directory to record results

        //call the planning and learning algorithms here
        //Uncomment which ever one(s) you want to run
        //example.BFSExample(outputPath);
        //example.DFSExample(outputPath);
        //example.AStarExample(outputPath);
        example.ValueIterationExample(outputPath);
        //example.PolicyIterationExample(outputPath);
        //example.QLearningExample(outputPath);
        //example.SarsaLearningExample(outputPath);
        //example.experimenterAndPlotter();

        //run the visualizer
        //example.visualize(outputPath);

        
    }

    //Define the BFS method
    public void BFSExample(String outputPath){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //BFS ignores reward. It just searches for a goal-condition-satisfying state.
        DeterministicPlanner planner = new BFS(domain, goalCondition, hashingFactory);
        planner.planFromState(initialState);

        //capture the computed plan in a partial policy
        Policy p = new SDPlannerPolicy(planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "BFS-results", sp);

    }

    //Depth-first search (DFS)
    public void DFSExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //DFS ignores reward. It just searches for a goal-condition-satisfying-state.
        DeterministicPlanner planner = new DFS(domain, goalCondition, hashingFactory);
        planner.planFromState(initialState);

        //capture the computed plan in a partial policy
        Policy p = new SDPlannerPolicy(planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "DFS-results", sp);

    }

    //A*
    public void AStarExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        Heuristic mdistHeuristic = new Heuristic() {

            @Override
            public double h(State s) {

                String an = GridWorldDomain.CLASSAGENT;
                String ln = GridWorldDomain.CLASSLOCATION;

                ObjectInstance agent = s.getObjectsOfTrueClass(an).get(0);
                ObjectInstance location = s.getObjectsOfTrueClass(ln).get(0);

                //get agent position
                int ax = agent.getDiscValForAttribute(GridWorldDomain.ATTX);
                int ay = agent.getDiscValForAttribute(GridWorldDomain.ATTY);

                //get location position
                int lx = location.getDiscValForAttribute(GridWorldDomain.ATTX);
                int ly = location.getDiscValForAttribute(GridWorldDomain.ATTY);

                 //compute Manhattan distance
                 double mdist = Math.abs(ax - lx) + Math.abs(ay - ly);

                 return -mdist;

            }
        };

        //Provide A* the heuristic and the reward function
        //So it can keep track of the actual cost
        DeterministicPlanner planner = new AStar(domain, rf, goalCondition, hashingFactory, mdistHeuristic);
        planner.planFromState(initialState);

        //capture the computed plan in a partial policy
        Policy p = new SDPlannerPolicy(planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "AStar-results", sp);

    }

    //Value Iteration
    public void ValueIterationExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //Time how long this takes
        long startTime = System.nanoTime();

        double gamma = 0.1;

        System.out.println("Gamma: " + gamma);
        OOMDPPlanner planner = new ValueIteration(domain, rf, tf, gamma, hashingFactory, 0.001, 100);
        planner.planFromState(initialState);

        long elapsedTime = System.nanoTime() - startTime;
        double elapsedTimeSeconds = elapsedTime/1000000000.0;
        double elapsedTimeSecondsRounded = (double)Math.round(elapsedTimeSeconds * 1000) / 1000;

        System.out.println("Value Iteration Running Time: " + elapsedTimeSecondsRounded + " seconds");

        //create a Q-greedy policy from the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "ValueIteration-results", sp);

        //visualize the value function and policy
        this.valueFunctionVisualize((QComputablePlanner)planner, p);

    }

    //Policy Iteration
    public void PolicyIterationExample(String outputPath){
        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //Time how long this takes
        long startTime = System.nanoTime();

        //Parameters: Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, StateHashFactory hashingFactory, double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
        OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, 0.99, hashingFactory, 0.001, 1000, 1000);
        planner.planFromState(initialState);

        long elapsedTime = System.nanoTime() - startTime;
        double elapsedTimeSeconds = elapsedTime/1000000000.0;
        double elapsedTimeSecondsRounded = (double)Math.round(elapsedTimeSeconds * 1000) / 1000;

        System.out.println("Policy Iteration Running Time: " + elapsedTimeSecondsRounded + " seconds");

        //create a Q-greedy policy from the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "PolicyIteration-results", sp);

        //visualize the value function and policy
        this.valueFunctionVisualize((QComputablePlanner)planner, p);

    }

    //Q-Learning
    public void QLearningExample(String outputPath){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //creating the learning algorithm object; discount = 0.99; initialQ=0.0; learning rate=0.9
        LearningAgent agent = new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);

        //run learning for 100 episodes
        for(int i = 0; i < 100; i++){
            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
            ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
            System.out.println(i + ": " + ea.numTimeSteps());
         }

    }

    //Sarsa(λ)
    public void SarsaLearningExample(String outputPath){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //discounts = 0.99; initialQ = 0.0; learning rate = 0.5; lambda = 1.0
        LearningAgent agent = new SarsaLam(domain, rf, tf, 0.99, hashingFactory, 0., 0.5, 1.0);

       //run learning for 100 episodes
       for(int i=0; i < 100; i++){
           EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
           ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
           System.out.println(i + ": " + ea.numTimeSteps());
       }

    }

    //Value function and policy visualization
    public void valueFunctionVisualize(QComputablePlanner planner, Policy p){
        List <State> allStates = StateReachability.getReachableStates(initialState, (SADomain)domain, hashingFactory);
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);

        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
        gui.setSpp(spp);
        gui.setPolicy(p);
        gui.setBgColor(Color.GRAY);
        gui.initGUI();

    }

    //Test the experimenter tools
    public void experimenterAndPlotter(){

        //custom reward function for more interesting results
        final RewardFunction rf = new GoalBasedRF(this.goalCondition, 5., -0.1);

        //Create factories for Q-learning agent and SARSA agent to compare
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
    
            @Override
            public String getAgentName() {
                return "Q-learning";
            }
    
            @Override
            public LearningAgent generateAgent() {
                return new QLearning(domain, rf, tf, 0.99, hashingFactory, 0.3, 0.1);
            }
        };

        LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

            @Override
            public String getAgentName() {
                return "SARSA";
            }

            @Override
            public LearningAgent generateAgent() {
                return new SarsaLam(domain, rf, tf, 0.99, hashingFactory, 0.0, 0.1, 1.);
            }
        };

        //Make a state generator that always returns the same initial state
        //Using the BURLAP-provided ConstantStateGenerator
        StateGenerator sg = new ConstantStateGenerator(this.initialState);

        //Create our experimenter, start it, and save all the data for all six metrics to CSV files.
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter((SADomain)this.domain, rf, sg, 10, 1000, qLearningFactory, sarsaLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
            TrialMode.MOSTRECENTANDAVERAGE,
            PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
            PerformanceMetric.AVERAGEEPISODEREWARD);

        exp.startExperiment();

        exp.writeStepAndEpisodeDataToCSV("expData");

    }
}


