package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{   
		/*Code Added */
		
		//Get all the states of the game from the 'policyValues'key set 
				Set<Game> statesAll = policyValues.keySet();
				
				//Initializing a new Random Object 
				Random rand =new Random();
				
				//Iterating through all the states of the game 
				for (Game state:statesAll)	{
					
					 // Check if the current state is not terminal
					if (!state.isTerminal()) {
						
						//Creating a list that contains all the possible move of the current state
						List <Move> movesPossible = state.getPossibleMoves();
						
						//check if the list is empty 
						if (!movesPossible.isEmpty()) {
							
							// Selecting a random move 'randMove' from the list 'movesPossible'
							Move randMove= movesPossible.get(rand.nextInt(movesPossible.size()));
							
							//Update the policy with the random move 
							this.curPolicy.put(state, randMove);
							
						}
						
					}
					
				}
				

	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{    /* Code Added */
		
		//Initializing maxQValue to a value grater than delta ,for the loop to be executed at least once 
				double maxQValue=delta+1;
				
				//iterating as long as the maximum change in Q value'maxQValue' is greater than 'delta'
				  while (maxQValue>delta) {
					  
					  //Initializing maxQValue to 'NEGATIVE_INFINITY'
					   maxQValue=Double.NEGATIVE_INFINITY;
					   
					   //Value iteration- for each state in the current policy key set 
					  for (Game state:curPolicy.keySet()){
						  
						  //Initializing the 'qVal' to 0
						  double qVal=0;
						  
						  //check if the  state is terminal 
						  if (!state.isTerminal()) {
							  
							  //Policy iteration Bellman's Equation 
							  for (TransitionProb t : mdp.generateTransitions(state, curPolicy.get(state))) {
			                      qVal += t.prob * (t.outcome.localReward + (discount * policyValues.get(t.outcome.sPrime)));
			                      	  
						  }
					  }
						  //Getting the  old Q value for the current state 
						  double oldQValue=policyValues.get(state);
						  
						  //updating the Q value for the current state in the policy 
						  policyValues.put(state, qVal);
						  
						  //Calcualting the maxQValue 
						  maxQValue=Math.max(maxQValue, Math.abs(oldQValue-qVal));
					  
				  }
					
					
					
					
				}

		
		
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{    /*Code Added */

		// Initializing the 'improved' to 'false' , check if the policy has improved 
		boolean improved=false;
		
		//Iterating through  all state of game in the current policy key set 
		for(Game state:curPolicy.keySet()) {
			
			//check if the  state is terminal 
			if (!state.isTerminal()) {
				
				//Initializing the 'maxQ' to 'NEGATIVE_INFINITY' ,lowest value 
				double maxQ=Double.NEGATIVE_INFINITY;
				
				//Initializing the 'optimalMove' to null
				Move optimalMove=null;
				
				//Iterating through all the possible  moves in the current state
				for (Move move :state.getPossibleMoves()) {
					
					//intialzing the 'qVal' to 0 
					double qVal=0;
					
					//Policy iteration Bellman's Equation 
					for (TransitionProb t : mdp.generateTransitions(state,move)) {
	                      qVal+= t.prob * (t.outcome.localReward + (discount * policyValues.get(t.outcome.sPrime)));	
				}
					//if the 'sumTotal' is greater than the 'maxQ' value then set then 'maxQ' value to the 'sumTotal' and 'optimalMove' to the current move 
					if (qVal>maxQ) {
						maxQ=qVal;
						optimalMove=move;
					}
				
			}
				
				// check if the 'optimalMove' is not null and not equals to the  current policy  
				if (optimalMove!=null && !optimalMove.equals(curPolicy.get(state))){
					
					//update the current policy to the optimal move 
					curPolicy.put(state, optimalMove);
					//set 'improved' to 'true'
					improved=true;
				}
					
				}
				
			
		}
		// return the improved policy 
		return improved;
		

	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{    /*Code Added */
		
		// Initializing  values 
				initValues();
				
				//Initializing random policies 
				initRandomPolicy();
				
				
				//Declaring previous policy
				HashMap<Game,Move> prevPolicy;
				
				// Evaluating the policy once before the loop 
				evaluatePolicy(delta);
				
				
				//Iterate till the policy is improved until convergence
				while(true) {
					
					//Evaluating the policy 
					evaluatePolicy(delta);
					
					//Storing the current policy before improvement 
					prevPolicy=new HashMap<>(this.curPolicy);
					
					// If the policy is not Improved then 'break' 
					if(!improvePolicy()) {
						break;
						
					}
					// If the current policy is equal to the previous policy  then 'break' 
					if (curPolicy.equals(prevPolicy)) {
						break;
					}
					
				}
				
				//Initializing a new policy 
				super.policy=new Policy();
				
				//setting new policy to  current policy
				super.policy.policy=curPolicy;
					
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
