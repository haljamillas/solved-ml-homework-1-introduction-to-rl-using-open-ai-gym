Download Link: https://assignmentchef.com/product/solved-ml-homework-1-introduction-to-rl-using-open-ai-gym
<br>
In this lab, we are required to work with Reinforcement Learning, a newer Machine Learning technique, that can train an agent in an environment.  Thea gent will navigate the classic 4×4 grid-world environment to a specific goal. The agent will learn an optimal 12policy through Q-Learning which will allow it to take actions to reach the goal while 13avoiding the boundaries. We use a platform here called AI gym to facilitate the whole 14process of the construction of the agent and the environment.

<h1>             Introduction</h1>

191.1                      <strong>What is Reinforcment Learning?</strong>

<ul>

 <li>Reinforcement Learning(RL) is one of the hottest research topics in the field of modern</li>

 <li>Artificial Intelligence and its popularity is only growing. Reinforcement Learning(RL) is 22        a type of machine learning technique that enables an agent to learn in an interactive 23            environment by trial and error using feedback from its own actions and experiences.</li>

 <li>Though both supervised and reinforcement learning use mapping between input and</li>

 <li>output, unlike supervised learning where the feedback provided to the agent is correct set 26 of actions for performing a task, reinforcement learning uses rewards and punishments as 27 signals for positive and negative behavior.</li>

 <li>As compared to unsupervised learning, reinforcement learning is different in terms of</li>

 <li>While the goal in unsupervised learning is to find similarities and differences</li>

 <li>between data points, in the case of reinforcement learning the goal is to find a suitable 31 action model that would maximize the total cumulative reward of the agent. The figure 32       below illustrates the action-reward feedback loop of a generic RL model.</li>

</ul>

<h1>               2.      Open AI Gym</h1>

<ul>

 <li>Gym is a toolkit for developing and comparing reinforcement learning algorithms. It</li>

 <li>makes no assumptions about the structure of your agent, and is compatible with any</li>

 <li>numerical computation library, such as TensorFlow or Theano.The <a href="https://github.com/openai/gym">gym</a> library is a</li>

 <li>collection of test problems — environments — that you can use to work out your 54 reinforcement learning algorithms. These environments have a shared interface, 55     allowing you to write general algorithms.</li>

</ul>

<h1>56                  3.       The Envrionment</h1>

<ul>

 <li>The Environment is a 4 by 4 grid environment described by two</li>

 <li>things the grid environment and the agent. An observation space</li>

 <li>which is defined as a vector of elements. This can be particularly</li>

 <li>useful for environments which return measurements, such as in 61 robotic environments.</li>

</ul>

<table width="34">

 <tbody>

  <tr>

   <td width="34">env</td>

  </tr>

 </tbody>

</table>

<table width="34">

 <tbody>

  <tr>

   <td width="34">env</td>

  </tr>

 </tbody>

</table>

62         The core gym interface is , which is the unified environment interface. The 63    following are the methods that would be quite helpful to us:

<table width="63">

 <tbody>

  <tr>

   <td width="63">env.reset</td>

  </tr>

 </tbody>

</table>

64: Resets the environment and returns a random initial state.

<table width="101">

 <tbody>

  <tr>

   <td width="101">env.step(action)</td>

  </tr>

 </tbody>

</table>

65: Step the environment by one timestep.

<ul>

 <li>Returns observation: Observations of the environment</li>

 <li>reward: If your action was beneficial or not</li>

 <li>done: Indicates if we have successfully picked up and dropped off a passenger, also</li>

 <li>called one <em>episode</em></li>

 <li>info: Additional info such as performance and latency for debugging purposes</li>

</ul>

<table width="72">

 <tbody>

  <tr>

   <td width="72">env.render</td>

  </tr>

 </tbody>

</table>

71: Renders one frame of the environment (helpful in visualizing the

<ul>

 <li>environment)</li>

 <li>We have an Action Space of size 4</li>

 <li>0 = down</li>

 <li>1 = up</li>

 <li>2 = right</li>

 <li>3 = left</li>

</ul>

<h1>78                  5.               The MULti-Bandit Problem (E-Greedy Algorithm)</h1>

<ul>

 <li>The multi-armed bandit problem is a classic reinforcement learning example where</li>

 <li>we are given a slot machine with <em>n</em> arms (bandits) with each arm having its own</li>

 <li>rigged probability distribution of success. Pulling any one of the arms gives you a</li>

 <li>stochastic reward of either R=+1 for success, or R=0 for failure. Our objective is to</li>

 <li>pull the arms one-by-one in sequence such that we maximize our total reward 84 collected in the long run.</li>

</ul>




<ul>

 <li><strong>The non-triviality of the multi-armed bandit problem lies in </strong></li>

 <li><strong>the fact that we (the agent) cannot access the true bandit </strong></li>

 <li><strong>probability distributions — all learning is carried out via the </strong>92 <strong>means of trial-and-error and value estimation. So the question </strong></li>

 <li><strong>is:</strong></li>

 <li>This is our goal for the multi-armed bandit problem, and having such a strategy 95 would prove very useful in many real-world situations where one would like to select 96               the “best” bandit out of a group of bandits.</li>

 <li>In this project, we approach the multi-armed bandit problem with a classical</li>

 <li>reinforcement learning technique of an <em>epsilon-greedy agent </em>with a learning framework 99 of <em>reward-average sampling </em>to compute the action-value Q(a) to help the agent improve 100         its future action decisions for long-term reward maximization.</li>

 <li>In a nutshell, the epsilon-greedy agent is a hybrid of a (1) completely-exploratory agent</li>

 <li>and a (2) completely-greedy agent. In the multi-armed bandit problem, a completely103 exploratory agent will sample all the bandits at a uniform rate and acquire knowledge</li>

 <li>about every bandit over time; the caveat of such an agent is that this knowledge is never</li>

 <li>utilized to help itself to make better future decisions! On the other extreme, a completely-</li>

 <li>greedy agent will choose a bandit and stick with its choice for the rest of eternity; it will 107 not make an effort to try out other bandits in the system to see whether they have better 108       success rates to help it maximize its long-term rewards, thus it is very narrow-minded!</li>

 <li>How do we perform this in our code?</li>

 <li>We perform this by assigning a variable called epsilon. This epsilon switches between the</li>

 <li>exploratory and the greedy agent. We choose a random number between 0 and 1, if this 112 number is less than epsilon, we tell the agent to explore if it is greater then we tell the 113          agent to be greedy. This tactic is used in our policy part of our code.</li>

</ul>

<h1>Q-Learning Algorithm</h1>

Essentially, Q-learning lets the agent use the environment’s rewards 117             to learn, over time, the best action to take in a given state.

<ul>

 <li>In our environment, we have the reward table,that the agent will learn from. It does</li>

 <li>thing by looking receiving a reward for taking an action in the current state, then 120 updating a <em>Q-value</em> to remember if that action was beneficial.</li>

 <li>The values store in the Q-table are called a <em>Q-values</em>, and they map to a (state,action)</li>

 <li></li>

 <li>A Q-value for a particular state-action combination is representative of the “quality” of 124 an action taken from that state. Better Q-values imply better chances of getting greater</li>

</ul>

125                rewards.




<ul>

 <li>Q-values are initialized to an arbitrary value, and as the agent exposes itself to the</li>

 <li>environment and receives different rewards by executing different actions, the Q129 values are updated using the equation:</li>

</ul>

130 <strong>Q(state,action)←(1−α)Q(state,action)+α(reward+γmaxaQ(nextα)Q(state,action)+α(reward+γmaxaQ(next)Q(state,action)+α(reward+γmaxaQ(nextα)Q(state,action)+α(reward+γmaxaQ(next(reward+α(reward+γmaxaQ(nextγmaxaQ(nextmaxaQ(next state,all actions))</strong>

131

<ul>

 <li>Where:</li>

 <li>– α(alpha) is the learning rate (0&lt;α≤1α≤1) – Just like in supervised learning settings, α is 134 the extent to which our Q-values are being updated in every iteration.</li>

 <li>– γ (gamma) is the discount factor (0≤γ≤1) – determines how much importance we</li>

 <li>want to give to future rewards. A high value for the discount factor (close to 1) 137 captures the long-term effective award, whereas, a discount factor of 0 makes our 138       agent consider only immediate reward, hence making it greedy.</li>

 <li>What is this saying?</li>

 <li>We are assigning (←), or updating, the Q-value of the agent’s current <em>state </em>and <em>action </em></li>

 <li>by first taking a weight (1−αα) of the old Q-value, then adding the learned value. The</li>

 <li>learned value is a combination of the reward for taking the current action in the current</li>

 <li>state, and the discounted maximum reward from the next state we will be in once we 144 take the current action.</li>

 <li>Basically, we are learning the proper action to take in the current state by looking at</li>

 <li>the reward for the current state/action combo, and the max rewards for the next state.</li>

 <li>This will eventually cause our taxi to consider the route with the best rewards strung</li>

 <li></li>

 <li>The Q-value of a state-action pair is the sum of the instant reward and the discounted</li>

 <li>future reward (of the resulting state). The way we store the Q-values for each state and</li>

 <li>action is through a Q-table</li>

 <li>Q-Table</li>

 <li>The Q-table is a matrix where we have a row for every state and a column for every 154          It’s first initialized to 0, and then values are updated after training.</li>

</ul>




<ul>

 <li>Epsilon</li>

 <li>We want the odds of the agent exploring to decrease as time goes</li>

 <li>One way to do this is by updated the epsilon every moment of</li>

 <li>the agent during the training phase. Choosing a decay rate was</li>

 <li>the difficult part. We want episilon not to decay too soon, so the</li>

 <li>agent can have time to explore. The way I went about this is I</li>

 <li>want the agent the ability to explore the region for at least the</li>

 <li>size of the grid movements. So in 25 movements the agent should 164 still have a good chance of being in exploratory phase. I chose</li>

 <li>decay = 1/1.01 because (1/1.01)^25 = 75% which is a good chance</li>

 <li>and still being in the exploratory phase within 0-25 moves. After 167 50 moves episilon goes to 0.6 which is good odds of agent doing 168           both exploring and action orientated.</li>

</ul>

169

<ul>

 <li>Results and Charts</li>

 <li>Here we plot epsilon vs episode.</li>

 <li>We see that we have a exponential decay going on here.</li>

</ul>

Then we plot rewards vs episode

Here we see that the total rewards slowly goes up then plateus after hitting 8. This is because 8 is the optimal path for our algrotihm.

CONCLUSION

Our Reinforcement Algorithm had done fairly well. Our agent has been able to learn the material in an effective manner. It has been able to reach its goal in 8 steps.