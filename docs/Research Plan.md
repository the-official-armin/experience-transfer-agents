**Research Question:** Can an open-source LLM agent use experiences from previous tasks to perform better on genuinely new tasks? If so, why does transfer work or fail, what makes an experience transferable, does representation of the experiences affect transfer, and how can we measure how transferable an experience is?

\*We wouldn’t be using RL because for an experiment like this we can just use external memory to use the experiences. RL can be a later experiment where we can bake the experiences into the model’s weights. 

**Hypothesis**: Experience is not equally transferable. The degree to which an experience helps on a new task depends on the properties and representation of the experience and on the type of shift between the previous and new task. 

- Simpler: Some of the experiences aren’t transferable for new tasks and some are. In fact, some can actually hurt. We would like to understand why.

Some key definitions to keep mind of during the research:

- Experience: Record of an agent’s interaction with a task. (task context, actions, observations, and outcome)  
- Transfer: Experience transfer occurs when the previous experience changes the agent's performance on a new task.  
  - Transfer Gain \= P(with experience)- P(without experience)  
  - If transfer gain is positive then there is a positive transfer and if the transfer gain is \- then there is a negative transfer (This is what makes experience and transfer more interesting, where can transfer be negative and hurtful to the agent?)  
- Novel Tasks: These are controlled task shifts that can provide us information on how the agent is doing on new tasks that it has not seen before. 

Experimental Setup: 

1. Create experience tasks.   
   1. Agent solves them  
   2. Record: trajectory, actions, observations, outcome  
2. Comparison: 2 conditions  
   1. Baseline: New task given, LLM agent answers, final outcome  
   2. Experience: new task, **retrieves relevant experience(this is the new part),** LLM agent, Answer  
   3. Then we calculate the transfer gain  
3. Answer the Research questions:  
   1. When and why does experience transfer or fail?  
      1. We measure transfer across the task specific hierarchy  
      2. Then investigate:   
- Was the new task given to the agent a success or failure?  
  - If failure was it because experience was too task-specific? wrong experience? Tool dependence? Overgeneralization?conflicting experiences? Or was it the inability to compose experiences?  
  2. What makes an experience transferable?  
     1. Which of the experience properties make an experience transferable  
        1. Positive?  
        2. Negative?  
        3. Neutral?  
     2. Examine Properties  
- Abstraction  
- Specificity  
- Task structure  
- Tool dependence  
- Length  
- Number of steps  
- success/faliure  
- Composability  
- Information Context  
  3. Does the representation of experience affect transfer?  
     1. Which representation works best?  
        1. Raw trajectory  
        2. Reflection  
        3. Procedure  
     2. We will try to calculate our performance with these 3 representations of experience  
  4. Can we measure or predict transferability?  
     1. Can we characterize how transferable an experience is?   
     2. Can we predict whether an experience will help a future task before actually using it? (**Biggest question\!**)

