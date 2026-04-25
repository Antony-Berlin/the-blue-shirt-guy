Theme #4 - Self-Improvement
The focus here is to create environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own capability growth. The objective is recursive skill amplification.
Expected Outcome: an environment for improving self-play of a LLM over a defined set of tasks
Example environments: Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

Judging Criteria

Criterion: Environment Innovation
Weight: 40%
What it means:
Is the environment novel, creative, or genuinely challenging?
Does it meaningfully test agent behavior in a way that hasn't been done before?


Criterion: Storytelling & Presentation
Weight: 30%
What it means:
Can you clearly explain the problem, the environment, and what the agent learned?
Is the demo engaging and easy to follow for a non-technical audience?


Criterion: Showing Improvement in Rewards
Weight: 20%
What it means:
Is there observable evidence of training progress? Reward curves, before/after behavior,
comparison against a baseline -- anything that proves the agent learned something.


Criterion: Reward & Training Pipeline
Weight: 10%
What it means:
Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained
agent's behavior?

Minimum Submission Requirements

NOTE: These are non-negotiable. Submissions missing any of these are at a serious disadvantage.
Use OpenEnv (latest release). Build on top of the framework; don’t reinvent the wheel.
A working training script using Unsloth or Hugging Face TRL, ideally as a Colab notebook so judges can re-run it.
Evidence that you actually trained; at minimum, loss and reward plots from a real run.
A short writeup: a mini-blog on Hugging Face or a < 2 minute video on YouTube explaining what your environment does and what you trained, or a short slide deck of presentation. Please make sure that all materials are linked from your README file so that judges can access them easily.


Engineer it cleanly (table stakes)
Engineering quality matters less than ambition, but sloppy work hurts. Make sure you:
Use OpenEnv’s Environment / MCPEnvironment base classes properly
Respect the client / server separation (clients should never import server internals)
Follow the standard Gym-style API (reset, step, state)
Have a valid openenv.yaml manifest
Don’t use reserved tool names (reset, step, state, close) for MCP tools