import burlap.domain.singleagent.gridworld.*;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

public class HelloGridWorld{

    public static void main(String [] args){

        GridWorldDomain gw = new GridWorldDomain(11,11); //Set grid world size
        gw.setMapToFourRooms(); //four rooms layout
        gw.setProbSucceedTransitionDynamics(0.8); //Stochastic transitions have an 0.8 success rate
        Domain domain = gw.generateDomain(); //generate the grid world domain

        //set up the initial state
        State s = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(s, 0, 0);
        GridWorldDomain.setLocation(s, 0, 10, 10);

        //Create visualizer and explorer
        Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
        VisualExplorer exp = new VisualExplorer(domain, v, s);

        //set control keys to use w-s-a-d
        exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
        exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
        exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
        exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);

        exp.initGUI();

    }

}
