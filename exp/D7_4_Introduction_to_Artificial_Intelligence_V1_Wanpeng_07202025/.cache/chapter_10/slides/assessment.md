# Assessment: Slides Generation - Week 10: Decision Making under Uncertainty: MDPs

## Section 1: Introduction to Decision Making under Uncertainty

### Learning Objectives
- Understand the importance of MDPs in decision-making.
- Recognize the challenges faced in uncertain environments.
- Differentiate between risk and uncertainty in decision-making contexts.
- Apply the MDP framework to model simple decision-making scenarios.

### Assessment Questions

**Question 1:** What is the primary concern of decision making under uncertainty?

  A) Maximizing rewards
  B) Predicting future outcomes
  C) Understanding risks
  D) Both A and C

**Correct Answer:** D
**Explanation:** Both maximizing rewards and understanding risks are crucial in decision making under uncertainty.

**Question 2:** In the context of MDPs, which component represents the possible situations?

  A) Actions
  B) States
  C) Transition Probabilities
  D) Rewards

**Correct Answer:** B
**Explanation:** States represent the various situations that can be encountered within the MDP framework.

**Question 3:** What differentiates risk from uncertainty in decision-making?

  A) Risk involves complete knowledge of outcomes.
  B) Uncertainty involves some knowledge of outcomes.
  C) Risk has known probabilities, while uncertainty does not.
  D) All of the above.

**Correct Answer:** C
**Explanation:** Risk is characterized by known probabilities associated with outcomes, while uncertainty is defined by the lack of such knowledge.

**Question 4:** Which of the following is an example of a decision-making scenario under uncertainty?

  A) Predicting stock prices
  B) Tossing a coin
  C) Rolling a die
  D) Playing a card game with no uncertainties

**Correct Answer:** A
**Explanation:** Predicting stock prices involves many unpredictable variables, making it a decision-making scenario under uncertainty.

### Activities
- In small groups, brainstorm and discuss real-life scenarios where decision-making is influenced by uncertainty. Identify at least three variables that contribute to the uncertainty in each scenario.

### Discussion Questions
- What are some strategies that can be employed to manage uncertainty in decision-making?
- How do MDPs improve the outcomes of decisions made under uncertainty?
- Can you think of a situation in your life where uncertainty played a significant role? How did you handle it?

---

## Section 2: What are Markov Decision Processes?

### Learning Objectives
- Define Markov Decision Processes.
- Identify and describe the components of MDPs, including states, actions, rewards, and transition probabilities.
- Illustrate the relationship between components of an MDP.

### Assessment Questions

**Question 1:** Which of the following is NOT a component of MDPs?

  A) States
  B) Actions
  C) Policies
  D) Forecasting

**Correct Answer:** D
**Explanation:** Forecasting is not a component of MDPs, which consist of states, actions, rewards, and transition probabilities.

**Question 2:** What represents the uncertainty of moving from one state to another in an MDP?

  A) Action Set
  B) State Space
  C) Transition Probabilities
  D) Reward Function

**Correct Answer:** C
**Explanation:** Transition probabilities capture the uncertainty associated with moving between states in response to actions.

**Question 3:** In a Markov Decision Process, what does the reward function represent?

  A) Probabilities of actions
  B) The goal of the agent
  C) Immediate benefits of actions
  D) Future predictions

**Correct Answer:** C
**Explanation:** The reward function provides a scalar value that represents the immediate benefit received from taking an action in a particular state.

**Question 4:** What is the purpose of the discount factor (γ) in the expected value formula of MDPs?

  A) To increase future rewards
  B) To emphasize immediate rewards over future rewards
  C) To eliminate uncertainty
  D) To measure state transitions

**Correct Answer:** B
**Explanation:** The discount factor (γ) is used to weigh immediate rewards more heavily than future rewards, reflecting how future rewards are valued.

### Activities
- Create a visual representation of MDP components by drawing a flowchart that includes states, actions, transition probabilities, and rewards.

### Discussion Questions
- How do Markov Decision Processes differ from traditional decision-making models?
- In what scenarios do you think MDPs are particularly useful?
- Can you think of real-world applications where MDPs could be implemented?

---

## Section 3: Components of MDPs

### Learning Objectives
- Explain the components of MDPs and their significance in decision-making.
- Understand the role of policies in influencing an agent's actions to maximize rewards.

### Assessment Questions

**Question 1:** What defines the long-term behavior of an agent in an MDP?

  A) Policy
  B) Action
  C) Reward
  D) Transition

**Correct Answer:** A
**Explanation:** The policy determines the long-term behavior of the agent by mapping states to actions.

**Question 2:** In a Markov Decision Process, which component represents feedback received after an action?

  A) State
  B) Reward
  C) Action
  D) Policy

**Correct Answer:** B
**Explanation:** The reward is a scalar feedback signal that indicates the immediate value of an action taken in a specific state.

**Question 3:** Which of the following is NOT a component of an MDP?

  A) State
  B) Variable
  C) Action
  D) Reward

**Correct Answer:** B
**Explanation:** A variable is not a component of an MDP. The components are state, action, reward, and policy.

**Question 4:** What does the transition function P(s'|s,a) define in an MDP?

  A) The probability of moving to state s' given the current state s and action a.
  B) The mapping of states to actions.
  C) The immediate reward received after taking action a.
  D) The agent's strategy in selecting actions based on states.

**Correct Answer:** A
**Explanation:** The transition function defines the probability of moving to state s' when taking action a in state s.

### Activities
- Create a hypothetical scenario using MDPs and define all four components: states, actions, rewards, and policies.
- Implement the components of an MDP in code and simulate an agent making decisions in a simple environment.

### Discussion Questions
- What are some real-world applications where MDPs could be effectively utilized?
- How might the choice of a policy affect the performance of an agent in a complex environment?

---

## Section 4: Understanding States and Actions

### Learning Objectives
- Differentiate between states and actions within MDPs.
- Assess how states and actions influence decision making.

### Assessment Questions

**Question 1:** In an MDP, what does the 'state' represent?

  A) Current situation of the agent
  B) Possible actions
  C) Reward received
  D) Policy applied

**Correct Answer:** A
**Explanation:** The 'state' represents the current situation of the agent in the environment.

**Question 2:** What is the role of an 'action' in an MDP?

  A) To transition between states
  B) To define rewards
  C) To create policies
  D) To estimate future states

**Correct Answer:** A
**Explanation:** The action is a decision that an agent can take when in a certain state, which impacts the subsequent state.

**Question 3:** How are transitions between states described in an MDP?

  A) Deterministic and fixed
  B) Probabilistic and based on actions
  C) Independent of actions
  D) Static and unrelated to states

**Correct Answer:** B
**Explanation:** Transitions are probabilistic and depend on the chosen action in a given state.

**Question 4:** What does a policy in an MDP define?

  A) The outcomes in each state
  B) The set of all states
  C) The actions to take in each state
  D) The rewards associated with each state

**Correct Answer:** C
**Explanation:** A policy defines the action that should be taken in each possible state.

### Activities
- Select a common board game and list out the states and potential actions available in the game.
- Create a simple flowchart that demonstrates the state transitions for a self-driving car based on various actions it can take.

### Discussion Questions
- Can you think of a situation in real life that can be modeled as an MDP? What would the states and actions be?
- How does the concept of exploration versus exploitation manifest in decision-making scenarios?

---

## Section 5: Rewards in MDPs

### Learning Objectives
- Understand the significance of rewards in decision making within MDPs.
- Analyze how rewards influence the agent's behavior and learning process.
- Implement reward structures in practical scenarios to improve decision-making strategies.

### Assessment Questions

**Question 1:** What is the primary role of rewards in a Markov Decision Process?

  A) To dictate the state transitions
  B) To evaluate the action taken
  C) To establish policies
  D) None of the above

**Correct Answer:** B
**Explanation:** Rewards evaluate the action taken and influence future decisions.

**Question 2:** What type of reward is immediately given after performing an action in a certain state?

  A) Cumulative Reward
  B) Expected Reward
  C) Immediate Reward
  D) Future Reward

**Correct Answer:** C
**Explanation:** The immediate reward is given right after the action in a state, reflecting the immediate benefit.

**Question 3:** What is the discount factor (γ) used for in reward calculations?

  A) To make future rewards more valuable
  B) To devalue immediate rewards
  C) To balance immediate and future rewards
  D) To identify terminal states

**Correct Answer:** C
**Explanation:** The discount factor helps in balancing between immediate and future rewards in the cumulative reward calculation.

**Question 4:** Which type of reward is designed to encourage exploration in reinforcement learning?

  A) Negative Reward
  B) Cumulative Reward
  C) Immediate Positive Reward
  D) Terminal Reward

**Correct Answer:** C
**Explanation:** Immediate positive rewards encourage exploration as agents are incentivized to seek out rewarding actions.

**Question 5:** How does a higher discount factor γ affect an agent's decision-making?

  A) It favors immediate rewards over future rewards.
  B) It ignores the importance of future rewards.
  C) It considers future rewards almost as valuable as immediate ones.
  D) It eliminates the role of immediate rewards.

**Correct Answer:** C
**Explanation:** A higher discount factor values future rewards more, thus promoting longer-term planning in decision-making.

### Activities
- Design a simple reward structure for an agent navigating a maze, considering both immediate and cumulative rewards.
- Experiment with different values of the discount factor (γ) in a reinforcement learning simulation, observing how it affects the agent's path to the goal.

### Discussion Questions
- How can poorly designed reward structures lead to unintended behaviors in agents?
- In what scenarios might it be beneficial to prioritize immediate rewards over future ones?

---

## Section 6: Transition Dynamics

### Learning Objectives
- Discuss the concept of transition dynamics in MDPs.
- Calculate transition probabilities for given states and actions.
- Articulate the importance of transition probabilities in dynamic programming and policy evaluation.

### Assessment Questions

**Question 1:** What do transition probabilities represent in the context of MDPs?

  A) Likelihood of receiving rewards
  B) Probability of moving to a new state after an action
  C) Determination of the current state
  D) Definition of policies

**Correct Answer:** B
**Explanation:** Transition probabilities indicate the likelihood of moving to a new state after taking a specific action.

**Question 2:** How is the expected immediate reward calculated in an MDP?

  A) R(s, a) = ∑ P(s' | s, a)
  B) R(s, a) = ∑ P(s' | s, a) × R(s, a, s')
  C) R(s, a) = P(s' | s, a) + R(s, a, s')
  D) R(s, a) = P(s' | s, a) - R(s, a, s')

**Correct Answer:** B
**Explanation:** The expected immediate reward is calculated by summing the product of transition probabilities and the rewards for each resultant state.

**Question 3:** What does the Markov property state in relation to transition dynamics?

  A) The next state is dependent on the previous states.
  B) Transition probabilities depend only on the current state and action.
  C) All actions lead to deterministic outcomes.
  D) States cannot transition back to previous states.

**Correct Answer:** B
**Explanation:** The Markov property states that transition probabilities are only dependent on the current state and the action taken, irrespective of previous states.

**Question 4:** In the example provided, what is the probability of slipping back to state (2, 2) when moving right?

  A) 0.7
  B) 0.2
  C) 0.1
  D) 1.0

**Correct Answer:** B
**Explanation:** In the example, the agent has a 0.2 probability of slipping back to state (2, 2) after attempting to move to the right.

### Activities
- Work in pairs to create a transition probability matrix for a simple MDP with three states and two possible actions. Discuss how the choices impact the transition probabilities.

### Discussion Questions
- How do transition probabilities impact the choice of action for an agent in uncertain environments?
- Can you think of real-world systems that exhibit similar transition dynamics? Share examples.
- What challenges do you see in estimating accurate transition probabilities in practical scenarios?

---

## Section 7: Policies in MDPs

### Learning Objectives
- Understand concepts from Policies in MDPs

### Activities
- Practice exercise for Policies in MDPs

### Discussion Questions
- Discuss the implications of Policies in MDPs

---

## Section 8: Goal of MDPs

### Learning Objectives
- Understand concepts from Goal of MDPs

### Activities
- Practice exercise for Goal of MDPs

### Discussion Questions
- Discuss the implications of Goal of MDPs

---

## Section 9: Solving MDPs

### Learning Objectives
- Identify common methods for solving MDPs.
- Understand the differences between Value Iteration and Policy Iteration.
- Apply the Bellman equation to compute state values.
- Analyze the convergence characteristics of both methods.

### Assessment Questions

**Question 1:** What is the primary purpose of solving an MDP?

  A) To find an optimal policy that maximizes cumulative rewards
  B) To minimize the number of states in the process
  C) To improve computational efficiency
  D) To calculate the transition probabilities

**Correct Answer:** A
**Explanation:** The primary purpose of solving MDPs is to find an optimal policy that maximizes cumulative rewards over time.

**Question 2:** Which equation is central to the Value Iteration process?

  A) Bellman equation
  B) Markowitz equation
  C) Law of total probability
  D) Central limit theorem

**Correct Answer:** A
**Explanation:** The Bellman equation is central to the Value Iteration process as it updates the value of each state based on possible actions and their outcomes.

**Question 3:** In Policy Iteration, what step comes after Policy Evaluation?

  A) Value Update
  B) Convergence Check
  C) Policy Improvement
  D) Initialization

**Correct Answer:** C
**Explanation:** In Policy Iteration, after evaluating the current policy, the next step is to improve the policy based on the evaluated values.

**Question 4:** Which method tends to require more computational resources per iteration?

  A) Value Iteration
  B) Policy Iteration
  C) Both are equally resource-intensive
  D) Neither, as they both minimize resources

**Correct Answer:** B
**Explanation:** Policy Iteration typically requires more computational resources per iteration than Value Iteration because it evaluates the entire policy before making improvements.

### Activities
- Implement a simple value iteration algorithm in Python. Create a grid world representation, define the reward structure, and use the Bellman update to compute state values until convergence.
- Design an MDP for a simple game scenario and apply the policy iteration method to find the optimal policy.

### Discussion Questions
- What scenarios might influence the choice between Value Iteration and Policy Iteration?
- How can the knowledge of MDP-solving methods be applied in real-world decision-making?
- Discuss potential limitations or challenges when implementing Value Iteration in large state spaces.

---

## Section 10: Value Iteration

### Learning Objectives
- Explain the steps involved in the value iteration process.
- Apply value iteration to a simple MDP example.
- Discuss the effects of different values for the discount factor on convergence and policy outcomes.

### Assessment Questions

**Question 1:** What is the primary concept behind value iteration?

  A) Evaluating the performance of policies
  B) Iteratively updating state values
  C) Transitioning between states
  D) Maximizing immediate rewards

**Correct Answer:** B
**Explanation:** Value iteration involves iteratively updating the values of states until convergence.

**Question 2:** Which of the following is NOT a component of an MDP?

  A) States (S)
  B) Actions (A)
  C) Value Estimates (V)
  D) Transition Probabilities (P)

**Correct Answer:** C
**Explanation:** Value Estimates (V) are derived from the MDP components but are not considered a fundamental part of an MDP.

**Question 3:** What role does the discount factor γ play in value iteration?

  A) It limits the number of iterations in the algorithm.
  B) It accounts for the uncertainty in rewards.
  C) It balances the trade-off between immediate and future rewards.
  D) It decides the order of state updates.

**Correct Answer:** C
**Explanation:** The discount factor γ, which ranges between 0 and 1, balances the importance of immediate rewards versus future rewards.

**Question 4:** What is the purpose of the policy extraction step in value iteration?

  A) To determine the transition probabilities
  B) To find the immediate rewards
  C) To derive the optimal action for each state based on the value function
  D) To initialize the value function

**Correct Answer:** C
**Explanation:** The policy extraction step identifies the optimal action for each state based on the computed value function.

### Activities
- Work in pairs to implement a small MDP using the value iteration algorithm based on given rewards and transition probabilities.
- Visualize the convergence of state values using a graph for iterative updates in a given MDP.

### Discussion Questions
- How does the choice of discount factor affect the policy derived from value iteration?
- Can you provide an example where value iteration may not be the best approach? Why?
- What challenges might arise when implementing the value iteration algorithm in a large-state space MDP?

---

## Section 11: Policy Iteration

### Learning Objectives
- Describe the process of policy iteration.
- Implement policy iteration on a given MDP.
- Analyze the convergence properties of the algorithm.

### Assessment Questions

**Question 1:** Which statement accurately describes policy iteration?

  A) It is slower than value iteration
  B) It consists of two main steps: policy evaluation and policy improvement
  C) It cannot be used with stochastic MDPs
  D) None of the above

**Correct Answer:** B
**Explanation:** Policy iteration consists of alternating between policy evaluation and policy improvement processes.

**Question 2:** What does the value function (V) represent in the context of policy iteration?

  A) The total reward achieved by an action
  B) The expected return from a state under a specific policy
  C) The probability of reaching a state
  D) The best action to take in a state

**Correct Answer:** B
**Explanation:** The value function provides the expected return from each state under a specific policy.

**Question 3:** What occurs during the policy improvement step in policy iteration?

  A) We update the value function V for the current policy.
  B) The policy is updated to choose actions that yield the highest value.
  C) We randomly select an action for all states.
  D) We check the convergence of the value function.

**Correct Answer:** B
**Explanation:** In the policy improvement step, the policy is updated to select actions that maximize expected returns.

**Question 4:** What is the role of initialization in the policy iteration process?

  A) It sets the action space for the agent.
  B) It defines the value of each state for every action.
  C) It provides a starting point for the policy and value function.
  D) It determines the structure of the Markov Decision Process.

**Correct Answer:** C
**Explanation:** Initialization provides a starting point for the policy and value function from which the algorithm iterates.

### Activities
- Work in pairs to solve a simple MDP using policy iteration. Define the states, actions, and transition probabilities, then implement the algorithm step by step.

### Discussion Questions
- How does the stochastic nature of MDPs affect the policy iteration algorithm?
- In what scenarios might policy iteration be preferred over value iteration?
- What challenges might arise when implementing policy iteration in large state spaces?

---

## Section 12: Applications of MDPs

### Learning Objectives
- Understand concepts from Applications of MDPs

### Activities
- Practice exercise for Applications of MDPs

### Discussion Questions
- Discuss the implications of Applications of MDPs

---

## Section 13: Challenges with MDPs

### Learning Objectives
- Discuss common challenges faced with Markov Decision Processes (MDPs).
- Evaluate potential solutions or alternatives to overcome MDP challenges.
- Analyze examples from real-world applications of MDPs to identify related challenges.

### Assessment Questions

**Question 1:** What is a common challenge when dealing with MDPs?

  A) Infinite state spaces
  B) Low-dimensional state spaces
  C) Fixed rewards
  D) Linear policies

**Correct Answer:** A
**Explanation:** Infinite or high-dimensional state spaces complicate the computational feasibility of MDPs.

**Question 2:** What does the curse of dimensionality imply in the context of MDPs?

  A) Fewer dimensions allow for better data collection.
  B) More dimensions require an exponentially larger amount of data for effective decision-making.
  C) Dimensionality does not affect the performance.
  D) It simplifies the data collection process.

**Correct Answer:** B
**Explanation:** The curse of dimensionality refers to the exponential increase in data needed as dimensions increase, complicating decision-making.

**Question 3:** How does non-stationarity affect MDP solutions?

  A) It ensures perfect foresight in predictions.
  B) It stabilizes the learned policies.
  C) It can make learned policies obsolete as conditions change.
  D) It has no effect on MDPs.

**Correct Answer:** C
**Explanation:** Non-stationarity can impact the stability of MDP solutions, rendering them less effective as conditions change.

**Question 4:** What is a primary computational challenge in large MDPs?

  A) Fixed size of state spaces
  B) Ability to predict all future outcomes
  C) Time complexity increases with state space size
  D) Lack of a reward structure

**Correct Answer:** C
**Explanation:** As the size of the state space increases, the time complexity for computing value functions and policies also increases.

### Activities
- Form small groups and brainstorm potential strategies to handle high-dimensional state spaces in MDPs. Present your ideas to the class.
- Select a specific application domain (e.g., healthcare, robotics, finance) and create a simplified MDP model with challenges noted in its implementation. Discuss potential solutions.

### Discussion Questions
- Can you think of a real-world scenario where non-stationarity could be detrimental to MDP performance? How would you address it?
- What techniques could be employed to reduce the impact of high-dimensional state spaces in MDP implementations?

---

## Section 14: Conclusion and Summary

### Learning Objectives
- Summarize the significance of MDPs in decision-making.
- Reflect on the lessons learned throughout the week.
- Identify real-world applications of MDPs in various fields.

### Assessment Questions

**Question 1:** What is a key takeaway regarding MDPs?

  A) They are less effective than other decision-making models
  B) They are versatile tools for uncertain decision-making
  C) Application is limited to AI
  D) None of the above

**Correct Answer:** B
**Explanation:** MDPs are versatile tools that can effectively aid in decision-making under uncertainty.

**Question 2:** What does the discount factor (γ) in an MDP signify?

  A) The penalty for making a wrong decision
  B) The importance of immediate rewards versus future rewards
  C) The total number of states
  D) The number of actions available

**Correct Answer:** B
**Explanation:** The discount factor (γ) represents how much value is placed on future rewards compared to immediate gains.

**Question 3:** Which of the following algorithms is NOT commonly used for solving MDPs?

  A) Value Iteration
  B) Policy Iteration
  C) Bellman Equation
  D) Gradient Descent

**Correct Answer:** D
**Explanation:** Gradient Descent is a general optimization technique not specifically tied to MDP solution methods like Value Iteration and Policy Iteration.

**Question 4:** In which of the following applications can MDPs be utilized?

  A) Predicting weather patterns
  B) Optimizing investment strategies
  C) Analyzing social media trends
  D) None of the above

**Correct Answer:** B
**Explanation:** MDPs can help model and optimize investment strategies by evaluating different states and actions over time.

### Activities
- Collaborate in small groups to summarize the key points discussed regarding MDPs this week, focusing on their applications and significance.

### Discussion Questions
- Consider a real-life decision-making scenario: How could you apply the concepts of MDPs to improve the decision-making process?
- What challenges might arise when implementing MDPs in a non-linear decision-making environment?

---

## Section 15: Discussion Questions

### Learning Objectives
- Encourage critical thinking regarding MDPs.
- Foster peer discussion and collaboration.
- Understand the key components and application of MDPs in decision-making.
- Identify challenges associated with using MDPs in complex environments.

### Assessment Questions

**Question 1:** What are the key components of a Markov Decision Process?

  A) States, Actions, Rewards
  B) States, Actions, Transition Probabilities, Rewards, Policy
  C) Actions, Rewards, Algorithms
  D) States, Algorithms, Policies

**Correct Answer:** B
**Explanation:** MDPs consist of states, actions, transition probabilities, rewards, and policies that guide decision-making in uncertain environments.

**Question 2:** How does an MDP handle uncertainty in decision-making?

  A) By eliminating all risks
  B) Through deterministic actions
  C) By using transition probabilities
  D) By simplifying the decision-making process

**Correct Answer:** C
**Explanation:** MDPs manage uncertainty by utilizing transition probabilities that capture the likelihood of moving from one state to another based on an action.

**Question 3:** What method is commonly used to find the optimal policy in an MDP?

  A) Linear Regression
  B) Value Iteration
  C) Gradient Descent
  D) Fourier Transform

**Correct Answer:** B
**Explanation:** Value Iteration is a method for computing the optimal policy by evaluating and improving policy estimates iteratively.

**Question 4:** Which of the following is a limitation of using MDPs?

  A) They always have a unique solution.
  B) They can be computationally intensive in large state spaces.
  C) They cannot be applied to real-world problems.
  D) They do not consider rewards.

**Correct Answer:** B
**Explanation:** In complex environments with large state spaces, MDPs can become computationally challenging, making them harder to solve.

### Activities
- In small groups, create a real-world example of a scenario where an MDP could apply effectively. Present your example and the reasoning behind your chosen states, actions, and rewards.

### Discussion Questions
- How does an MDP facilitate decision-making when considering both immediate rewards and long-term benefits?
- Which industries can benefit from MDPs, and what specific scenarios illustrate their application?
- What methods, such as Value Iteration and Policy Iteration, can be used to determine the optimal policy within an MDP?

---

## Section 16: Further Reading and Resources

### Learning Objectives
- Identify additional resources for further study of MDPs.
- Formulate a plan for self-directed learning based on the provided readings.

### Assessment Questions

**Question 1:** What is one of the main components of a Markov Decision Process (MDP)?

  A) Action Space
  B) Configuration Space
  C) Solution Space
  D) Utility Space

**Correct Answer:** A
**Explanation:** One of the main components of an MDP is the Action Space, which is the set of actions that can be taken.

**Question 2:** Which book emphasizes the relationship between dynamic programming and optimal decision-making?

  A) Markov Decision Processes: Algorithms and Applications
  B) Dynamic Programming and Optimal Control
  C) Reinforcement Learning: An Introduction
  D) A Survey of Markov Decision Process Applications

**Correct Answer:** B
**Explanation:** The book 'Dynamic Programming and Optimal Control' by Dimitri P. Bertsekas emphasizes the connection between dynamic programming and optimal decision-making.

**Question 3:** What is the discount factor (γ) in an MDP used for?

  A) To calculate the transition probabilities
  B) To prioritize immediate rewards over future rewards
  C) To define the set of actions
  D) To determine the set of states

**Correct Answer:** B
**Explanation:** The discount factor (γ) is used in MDPs to prioritize immediate rewards over future rewards, affecting how decisions are made.

**Question 4:** Which of the following is a practical tool for developing reinforcement learning algorithms?

  A) TensorFlow
  B) OpenAI Gym
  C) PyTorch
  D) NumPy

**Correct Answer:** B
**Explanation:** OpenAI Gym is a toolkit specifically designed for developing and comparing reinforcement learning algorithms.

### Activities
- Explore the recommended resources and summarize key insights from each one.
- Create a brief presentation on how MDPs can be applied in a specific industry of your choice.

### Discussion Questions
- How do you think MDPs can be applied to solve real-world decision-making problems?
- Discuss the relationship between MDPs and reinforcement learning. How do they complement each other?

---

