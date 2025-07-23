# Assessment: Slides Generation - Week 2: Markov Decision Processes (MDPs)

## Section 1: Introduction

### Learning Objectives
- Define the components of a Markov Decision Process and explain their significance.
- Describe the Bellman Equation and its role in solving MDPs.
- Apply the concepts of MDPs to solve a decision-making problem, creating a simple model.

### Assessment Questions

**Question 1:** What does the 'P' in an MDP stand for?

  A) Reward Function
  B) Transition Probability Function
  C) Discount Factor
  D) Action Set

**Correct Answer:** B
**Explanation:** The 'P' in an MDP stands for the transition probability function, which defines the probability of moving from one state to another given a specific action.

**Question 2:** Which factor reflects the importance of future rewards in an MDP?

  A) S
  B) A
  C) R
  D) γ

**Correct Answer:** D
**Explanation:** The discount factor 'γ' indicates how much a decision-maker values future rewards compared to immediate rewards, with a value between 0 and 1.

**Question 3:** In the Bellman Equation, what does V(s) represent?

  A) The maximum expected reward from all states
  B) The current state of the process
  C) The value function representing the maximum expected reward from a state
  D) The action taken at state s

**Correct Answer:** C
**Explanation:** In the Bellman Equation, V(s) represents the value function that indicates the maximum expected reward obtainable from state 's' by following the optimal policy.

**Question 4:** What is the key application of MDPs?

  A) Financial forecasting
  B) Decision-making under uncertainty
  C) Linear regression
  D) Statistical hypothesis testing

**Correct Answer:** B
**Explanation:** MDPs are primarily used for modeling decision-making scenarios where the outcomes are uncertain and depend on both probabilistic events and deterministic choices.

### Activities
- Create a simple grid world representation and identify the states, actions, transition probabilities, and rewards assigned to each action in the grid.
- Using the definitions provided, construct a basic MDP for a scenario of your choice, and describe each component (states, actions, transition probabilities, rewards, and discount factor).

### Discussion Questions
- How would the concept of MDPs apply to real-world decision-making scenarios, such as autonomous navigation or resource allocation?
- What challenges do you think arise when implementing MDPs in complex systems with many states and actions?

---

## Section 2: Overview

### Learning Objectives
- Understand concepts from Overview

### Activities
- Practice exercise for Overview

### Discussion Questions
- Discuss the implications of Overview

---

## Section 3: Conclusion

### Learning Objectives
- Understand the key components of Markov Decision Processes and their significance in decision-making.
- Apply the concepts of MDPs to practical scenarios and develop algorithms to solve MDPs.

### Assessment Questions

**Question 1:** What is the primary goal of a Markov Decision Process (MDP)?

  A) To minimize costs
  B) To find the optimal policy maximizing expected rewards
  C) To model deterministic processes
  D) To analyze historical data

**Correct Answer:** B
**Explanation:** The primary goal of an MDP is to find a policy that maximizes the expected sum of rewards over time.

**Question 2:** Which of the following best describes the transition function in an MDP?

  A) It specifies the immediate reward for actions
  B) It defines the states available to the decision-maker
  C) It describes probabilities of moving between states given an action
  D) It measures the future impact of current rewards

**Correct Answer:** C
**Explanation:** The transition function, typically denoted as P, describes the probabilities of moving from one state to another after taking an action.

**Question 3:** What role does the discount factor (γ) play in an MDP?

  A) It determines the number of states in a process
  B) It indicates the immediate reward for actions taken
  C) It weighs future rewards relative to immediate rewards
  D) It is used to measure the entropy of the states

**Correct Answer:** C
**Explanation:** The discount factor (γ) represents the importance of future rewards as compared to immediate rewards, with values ranging from 0 to 1.

**Question 4:** In the grid world example, what would be a possible reward for reaching a goal state?

  A) 0
  B) -10
  C) +10
  D) +5

**Correct Answer:** C
**Explanation:** In the grid world example, the agent receives a positive reward (+10) for reaching a goal state.

### Activities
- Implement a simple grid world MDP in Python. Define states, actions, transition functions, and a reward structure. Then use Value Iteration to calculate the optimal policy.
- Create a table summarizing different real-world applications of MDPs, detailing the states, actions, transition functions, and rewards in each case.

### Discussion Questions
- How would the concept of MDPs change if the discount factor (γ) were set to 0 versus 1?
- Can you think of any limitations or challenges when applying MDPs to real-world decision-making problems?

---

