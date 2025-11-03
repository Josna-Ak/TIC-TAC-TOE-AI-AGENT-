package ticTacToe;

import java.util.List;
import java.util.Random;


/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()

	{
		
		this(new RandomAgent(), 0.1, 50000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		/* Code Added */
		
		//Initializing a random variable 
		Random rand= new Random();
		// Iterating over each episode to initialize the training process
				for (int ep = 0; ep < numEpisodes; ep++) {

					// Resetting the environment for the new episode
					env.reset();

					// Get the current game state
					Game currGame = env.getCurrentGameState();

					// Check if the game reached the terminal state
					while (!env.isTerminal()) {	
						//Initializing the move to be selected to 'null'
						Move selectMove = null;

						// Deciding if  to explore or exploit
						if (rand.nextDouble() < epsilon) { 
							// exploration

							// Initialize a list that stores  all that possible moves
							List<Move> possibleMoves = env.getPossibleMoves();

							// Select a random move for exploration
							selectMove = possibleMoves.get(rand.nextInt(possibleMoves.size()));

						} else { 
							// exploitation
							//Initializing highest Q value to 'NEGATIVE_INFINITY'
							double highestQVal = Double.NEGATIVE_INFINITY;

							// Iterating over possible moves to find the highest Q-value
							for (Move validMove : env.getPossibleMoves()) {

								// Getting the Q-value for the move in the current game  
								double validQVal = qTable.getQValue(currGame, validMove);

								//if the 'validQVal' is greater than the 'highestQVal' value then set
								//then it value to 'validQVal' and 'selectMove' to the 'validMove'
								if (validQVal > highestQVal) {
									highestQVal = validQVal;
									selectMove = validMove;
								}
							}
						}
						//Play the game ,and catch the error for illegal move 

						try {
							// Initializing outcome that execute the selected move 
							Outcome outcome = env.executeMove(selectMove);

							// saving the state of the current game 
							final Game savedState = currGame;
							
							

							// Initializing the maximum Q value  to 'NEGATIVE_INFINITY'
							double maxQVal = Double.NEGATIVE_INFINITY;
							
							//Iterating oner the moves of the  current state 
							for (Move m : savedState.getPossibleMoves()) {
								
								//Initializing the 'qVal' to get the Q value from the qTable 
								double qVal = qTable.getQValue(savedState, m);
								
								// Updating the 'maxQVal' to get the greater Q value
								maxQVal = Math.max(qVal, maxQVal);
							}

							// Declaring the sample for the Q-value update
							double sample;
							
							
							//check if the saved state is terminal
							if (savedState.isTerminal()) {
								
								//Set the sample to store the outcome.localRewards 
								sample = outcome.localReward;
							}
							else {
								
								//Iterate for all the moves for current state
								for (Move m : savedState.getPossibleMoves()) {
									//Initializing qVal to get the Q value from the Q table 
									double qVal = qTable.getQValue(savedState, m);
									//set maxQVal to QVal if it is greater 
									if (qVal > maxQVal) {
										maxQVal = qVal;
									}
								}
								//set the sample to get the future reward 
								sample = outcome.localReward + discount * maxQVal;
							}

							// Compute the updated Q-value using the Q-learning formula
							double qValUpdate = (1 - alpha) * qTable.getQValue(outcome.s, outcome.move) + alpha * sample;

							// Updating the Q-value in the Q-table
							qTable.addQValue(outcome.s, outcome.move, qValUpdate);

							// Updating the current game state
							currGame = outcome.sPrime;
						} catch (IllegalMoveException e) {

							// Print the Exception message
							System.out.print("Illegal Move!!!!!!!!");
						}
					}
				}

		
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		/*Code Added*/
		
		// Creating a new  Policy object
	    Policy p = new Policy();

	    // For each 'state' in the key set of 'valueFunction'
	    for (Game state : qTable.keySet()) {

	        // Checking if the current state is not terminal
	        if (!state.isTerminal()) {

	            // Initializing 'maxQval' to negative infinity
	            double maxQVal = Double.NEGATIVE_INFINITY;

	            // Initializing 'optimalMove' to null
	            Move optimalMove = null;

	            // Iterating through each possible 'm' in the current 'state'
	            for (Move m : state.getPossibleMoves()) {

	                // Initializing 'qVal' to 0
	                double qVal = qTable.getQValue(state, m);

	                // If 'qValue' is greater than 'maxQ', updating 'maxQ' to be 'qValue' and updating
	                // 'optimalMove' to be the current 'm'
	                if (qVal > maxQVal) {
	                    maxQVal = qVal;
	                    optimalMove = m;
	                }
	            }

	            // Updating the policy of the current 'state' to be 'optimalMove'
	            p.policy.put(state, optimalMove);
	        }
	    }

	    // Returning the Optimal Policy
	    return p;
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
