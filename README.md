Download Link: https://assignmentchef.com/product/solved-cs285-assignment-2-policy-gradients
<br>
The goal of this assignment is to experiment with policy gradient and its variants, including variance reduction tricks such as implementing reward-to-go and neural network baselines. The startercode can be found at <a href="https://github.com/berkeleydeeprlcourse/homework_fall2019/tree/master/hw2">https://github.com/berkeleydeeprlcourse/ </a><a href="https://github.com/berkeleydeeprlcourse/homework_fall2019/tree/master/hw2">homework_fall2019/tree/master/hw2</a><a href="https://github.com/berkeleydeeprlcourse/homework_fall2019/tree/master/hw2">.</a>

<h1>2      Review</h1>

<h2>2.1       Policy gradient</h2>

Recall that the reinforcement learning objective is to learn a <em>θ</em><sup>∗ </sup>that maximizes the objective function:

<em>J</em>(<em>θ</em>) = E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>r</em>(<em>τ</em>)]                                                         (1)

where each rollout <em>τ </em>is of length <em>T</em>, as follows:

<em>T</em>−1

<em>π</em><em>θ</em>(<em>τ</em>) = <em>p</em>(<em>s</em>0<em>,a</em>0<em>,…,s</em><em>T</em>−1<em>,a</em><em>T</em>−1) = <em>p</em>(<em>s</em>0)<em>π</em><em>θ</em>(<em>a</em>0|<em>s</em>0) Y <em>p</em>(<em>s</em><em>t</em>|<em>s</em><em>t</em>−1<em>,a</em><em>t</em>−1)<em>π</em><em>θ</em>(<em>a</em><em>t</em>|<em>s</em><em>t</em>)

<em>t</em>=1

and

<em>T</em>−1

<em>r</em>(<em>τ</em>) = <em>r</em>(<em>s</em><sub>0</sub><em>,a</em><sub>0</sub><em>,…,s<sub>T</sub></em><sub>−1</sub><em>,a<sub>T</sub></em><sub>−1</sub>) = <sup>X</sup><em>r</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>)<em>.</em>

<em>t</em>=0

The policy gradient approach is to directly take the gradient of this objective:

<table width="439">

 <tbody>

  <tr>

   <td width="419">Z∇<em><sub>θ</sub>J</em>(<em>θ</em>) = ∇<em><sub>θ                    </sub>π<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)<em>dτ</em></td>

   <td width="20">(2)</td>

  </tr>

  <tr>

   <td width="419">Z=        <em>π<sub>θ</sub></em>(<em>τ</em>)∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)<em>dτ.</em></td>

   <td width="20">(3)</td>

  </tr>

  <tr>

   <td width="419">= E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)]</td>

   <td width="20">(4)</td>

  </tr>

 </tbody>

</table>

(5)

In practice, the expectation over trajectories <em>τ </em>can be approximated from a batch of <em>N </em>sampled trajectories:

)                                                                 (6)

<em>.                        </em>(7)

Here we see that the policy <em>π<sub>θ </sub></em>is a probability distribution over the action space, conditioned on the state. In the agent-environment loop, the agent samples an action <em>a<sub>t </sub></em>from <em>π<sub>θ</sub></em>(·|<em>s<sub>t</sub></em>) and the environment responds with a reward <em>r</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>).

<h2>2.2        Variance Reduction</h2>

<strong>2.2.1      Reward-to-go</strong>

One way to reduce the variance of the policy gradient is to exploit causality: the notion that the policy cannot affect rewards in the past. This yields the following modified objective, where the sum of rewards here does not include the rewards achieved prior to the time step at which the policy is being queried. This sum of rewards is a sample estimate of the <em>Q </em>function, and is referred to as the “reward-to-go.”

<em> .                          </em>(8)

<strong>2.2.2      Discounting</strong>

Multiplying a discount factor <em>γ </em>to the rewards can be interpreted as encouraging the agent to focus more on the rewards that are closer in time, and less on the rewards that are further in the future. This can also be thought of as a means for reducing variance (because there is more variance possible when considering futures that are further into the future). We saw in lecture that the discount factor can be incorporated in two ways, as shown below.

The first way applies the discount on the rewards from full trajectory:

!

(9)

and the second way applies the discount on the “reward-to-go:”

<em> .                     </em>(10)

.

<strong>2.2.3     Baseline</strong>

Another variance reduction method is to subtract a baseline (that is a constant with respect to <em>τ</em>) from the sum of rewards:

∇<em><sub>θ</sub>J</em>(<em>θ</em>) = ∇<em><sub>θ</sub></em>E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>r</em>(<em>τ</em>) − <em>b</em>]<em>.                                                </em>(11)

This leaves the policy gradient unbiased because

∇<em><sub>θ</sub></em>E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>b</em>] = E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>τ</em>) · <em>b</em>] = 0<em>.</em>

In this assignment, we will implement a value function <em>V<sub>φ</sub><sup>π </sup></em>which acts as a <em>state-dependent </em>baseline. This value function will be trained to approximate the sum of future rewards starting from a particular state:

<em>T</em>−1

<em>V<sub>φ</sub><sup>π</sup></em>(<em>s<sub>t</sub></em>) ≈ <sup>X</sup>E<em>π<sub>θ </sub></em>[<em>r</em>(<em>s<sub>t</sub></em>0<em>,a<sub>t</sub></em>0)|<em>s<sub>t</sub></em>]<em>,                                                 </em>(12)

<em>t</em><sup>0</sup>=<em>t</em>

so the approximate policy gradient now looks like this:

<em> .         </em>(13)

<h1>3        Overview of Implementation</h1>

<h2>3.1      Files</h2>

To implement policy gradients, we will be building up the code that we started in homework 1. To clarify, all files needed to run your code are in this new repository, but if you look through the code, you’ll see that it’s building on top of the code structure/hierarchy that we laid out in the previous assignment.

You will need to get some portions of code from your homework 1 implementation, and copy them into this new homework 2 repository. These parts are marked in the code with

# TODO: GETTHIS from HW1. Note that solutions for HW1 will be released on Piazza, so you can ensure that your implementation is correct. The following files have these placeholders that you need to fill in by copying code from HW1:

<ul>

 <li>infrastructure/tfutils.py</li>

 <li>infrastructure/utils.py</li>

 <li>infrastructure/rltrainer.py</li>

 <li>policies/MLPpolicy.py</li>

</ul>

After bringing in the required components from the previous homework, you can then begin to work on the code needed for this assignment. These placeholders are marked with TODO, and they can be found in the following files for you to fill out:

<ul>

 <li>agents/pgagent.py</li>

 <li>policies/MLPpolicy.py</li>

</ul>

Similar to the previous homework assignment, the script to run is found inside the scripts directory, and the commands to run are included in the README.

<h2>3.2       Overview</h2>

As in the previous homework, the main training loop is implemented in infrastructure/rltrainer.py.

The policy gradient algorithm uses the following 3 steps:

<ol>

 <li><em>Sample trajectories </em>by generating rollouts under your current policy.</li>

 <li><em>Estimate returns and compute advantages</em>. This is executed in the train function of py</li>

 <li><em>Train/Update parameters</em>. The computational graph for the policy and the baseline, as well as the update functions, are implemented in policies/MLPpolicy.py.</li>

</ol>

<h1>4         Implementing Vanilla Policy Gradients</h1>

We start by implementing the simplest form of policy gradient algorithms (vanilla policy gradient), as given by equation 9. In this section and the next one, you can ignore blanks in the code that are used if self.nnbaseline is set to true.

<strong>Problem 1</strong>

<ol>

 <li>Copy code from HW1 to fill in the blanks, as indicated by # TODO: GETTHIS from HW1 in the following files:</li>

</ol>

infrastructure/tfutils.py infrastructure/utils.py infrastructure/rltrainer.py

<ol start="2">

 <li>Read and fill in the # TODO blanks in the train function in py</li>

 <li>Implement estimating the return by filling in the blanks in the calculateqvals function (also in py). Use the discounted return for the full trajectory:</li>

</ol>

<em>T</em>−1 <em>r</em>(<em>τ</em><em>i</em>) = X<em>γ</em><em>t</em>0<em>r</em>(<em>s</em><em>it</em>0<em>,a</em><em>it</em>0)

<em>t</em><sup>0</sup>=0

Note that this means you only need to fill in “Case 1” inside this function (under if not self.rewardtogo).

<ol start="4">

 <li>Copy code from HW1 to fill in some blanks, as indicated by # TODO: GETTHIS from HW1 in policies/MLPpolicy.py</li>

 <li>Define the computational graph for the policy (definetrainop) in policies/MLPpolicy.py</li>

</ol>

<h1>5        Implementing “reward-to-go”</h1>

<strong>Problem 2 </strong>Implement the “reward-to-go” formulation (Equation 10) for estimating the q value. Here, the returns are estimated as follows:

<em>T</em>−1

<em>r</em>(<em>τ</em><em>i</em>) = X<em>γ</em><em>t</em>0−<em>t</em><em>r</em>(<em>s</em><em>it</em>0<em>,a</em><em>it</em>0)                                                      (14)

<em>t</em><sup>0</sup>=<em>t</em>

Note that this is “Case 2” inside the calculateqvals function in pgagent.py).

<h1>6         Experiments: Policy Gradient</h1>

After you have implemented the code from sections 4 and 5 above, you will run experiments to get a feel for how different settings impact the performance of policy gradient methods.

<strong>Problem 3. CartPole: </strong>Run multiple experiments with the PG algorithm on the discrete CartPole-v0 environment, using the following commands:

<table width="624">

 <tbody>

  <tr>

   <td width="624">python run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 1000 -dsa–exp_name sb_no_rtg_dsapython run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 1000 -rtg dsa –exp_name sb_rtg_dsapython run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 1000 -rtg–exp_name sb_rtg_napython run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 5000 -dsa–exp_name lb_no_rtg_dsapython run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 5000 -rtg dsa –exp_name lb_rtg_dsapython run_hw2_policy_gradient.py –env_name CartPole-v0 -n 100 -b 5000 -rtg–exp_name lb_rtg_na</td>

  </tr>

 </tbody>

</table>

What’s happening there:

<ul>

 <li>-n : Number of iterations.</li>

 <li>-b : Batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).</li>

 <li>-dsa : Flag: if present, sets standardize_advantages to False. Otherwise, by default, standardize_advantages=True.</li>

 <li>-rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.</li>

 <li>–exp_name : Name for experiment, which goes into the name for the data logging directory.</li>

</ul>

Various other command line arguments will allow you to set batch size, learning rate, network architecture (number of hidden layers and the size of the hidden layers—for CartPole, you can use one hidden layer with 32 units), and more. You can change these as well, but keep them <strong>FIXED </strong>between the 6 experiments mentioned above.

<strong>Deliverables for report:</strong>

<ul>

 <li>Create two graphs:

  <ul>

   <li>In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with sb_. (The small batch experiments.)</li>

   <li>In the second graph, compare the learning curves for the experiments prefixed with lb_. (The large batch experiments.)</li>

  </ul></li>

 <li>Answer the following questions briefly:

  <ul>

   <li>Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?</li>

   <li>Did advantage standardization help?</li>

   <li>Did the batch size make an impact?</li>

  </ul></li>

 <li>Provide the exact command line configurations you used to run your experiments. (To verify your chosen values for the other parameters, such as learning rate, architecture, and so on.)</li>

</ul>

<strong>What to Expect:</strong>

<ul>

 <li>The best configuration of CartPole in both the large and small batch cases should converge to a maximum score of 200.</li>

</ul>

<strong>Problem 4. InvertedPendulum: </strong>Run experiments in InvertedPendulum-v2 continuous control environment as follows:

python run_hw2_policy_gradient.py –env_name InvertedPendulum-v2 –ep_len 1000 –discount 0.9 -n 100 -l 2 -s 64 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –exp_name ip_b&lt;b*&gt;_r&lt;r*&gt;

where your task is to find the smallest batch size b* and largest learning rate r* that gets to optimum (maximum score of 1000) in less than 100 iterations. The policy performance may fluctuate around 1000: This is fine. The precision of b* and r* need only be one significant digit.

<strong>Deliverables:</strong>

<ul>

 <li>Given the b* and r* you found, provide a learning curve where the policy gets to optimum (maximum score of 1000) in less than 100 iterations. (This may be for a single random seed, or averaged over multiple.)</li>

 <li>Provide the exact command line configurations you used to run your experiments.</li>

</ul>

<h1>7         Implementing Neural Network Baseline</h1>

For the rest of the assignment we will use “reward-to-go.”

<strong>Problem 5. </strong>We will now implement a value function as a state-dependent neural network baseline. Note that there is nothing to submit for this problem, but subsequent sections will require this code.

<ol>

 <li>In py implement, a neural network that predicts the expected return conditioned on a state. Also implement the loss function to train this network and its update operation self.baselineop.</li>

 <li>In estimate_advantage in pg_agent.py, use the neural network to predict the expected state-conditioned return, standardize it to match the statistics of the current batch of “reward-to-go”, and subtract this value from the “reward-to-go” to yield an estimate of the advantage. Follow the hints and guidelines in the code. This implements</li>

 <li>In update, update the parameters of the the neural network baseline by using the Tensorflow session to call self.baselineop. “Rescale” the target values for the neural network baseline to have a mean of zero and a standard deviation of one. You should now have completed all # TODO entries in this file.</li>

</ol>

<h1>8         Experiments: More Complex Tasks</h1>

<strong>Note: </strong>The following tasks take quite a bit of time to train. Please start early!

<strong>Problem 6: LunarLander </strong>For this problem, you will use your policy gradient implementation to solve LunarLanderContinuous-v2. Use an episode length of 1000. Note that depending on how you installed gym, you may need to run pip install ’gym[all]’ in order to use this environment. The purpose of this problem is to test and help you debug your baseline implementation from Section 7 above.

Run the following command:

<table width="624">

 <tbody>

  <tr>

   <td width="624">python run_hw2_policy_gradient.py –env_name LunarLanderContinuous-v2 –ep_len1000 –discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg          -nn_baseline –exp_name ll_b40000_r0.005}</td>

  </tr>

 </tbody>

</table>

<strong>Deliverables:</strong>

<ul>

 <li>Plot a learning curve for the above command. You should expect to achieve an average return of around 180.</li>

</ul>

<strong>Problem 7: HalfCheetah </strong>For this problem, you will use your policy gradient implementation to solve HalfCheetah-v2. Use an episode length of 150, which is shorter than the default of 1000 for HalfCheetah (which would speed up your training significantly). Search over batch sizes b ∈ [10000<em>,</em>30000<em>,</em>50000] and learning rates r ∈ [0<em>.</em>005<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>02] to replace &lt;b&gt; and &lt;r&gt; below.

python run_hw2_policy_gradient.py –env_name HalfCheetah-v2 –ep_len 150 -discount 0.95 -n 100 -l 2 -s 32 -b &lt;b&gt; -lr &lt;r&gt; –video_log_freq -1 -reward_to_go –nn_baseline –exp_name hc_b&lt;b&gt;_lr&lt;r&gt;_nnbaseline

<strong>Deliverables:</strong>

<ul>

 <li>Provide a single plot with the learning curves for the HalfCheetah experiments that you tried. Also, describe in words how the batch size and learning rate affected task performance.</li>

</ul>

Once you’ve found suitable values of b and r among those choices (let’s call them b* and r*), use b* and r* and run the following commands (remember to replace the terms in the angle brackets):

<table width="624">

 <tbody>

  <tr>

   <td width="624">python run_hw2_policy_gradient.py –env_name HalfCheetah-v2 –ep_len 150 -discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt;python run_hw2_policy_gradient.py –env_name HalfCheetah-v2 –ep_len 150 -discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt;python run_hw2_policy_gradient.py –env_name HalfCheetah-v2 –ep_len 150 -discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; –nn_baseline –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt;python run_hw2_policy_gradient.py –env_name HalfCheetah-v2 –ep_len 150 -discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –nn_baseline -exp_name hc_b&lt;b*&gt;_r&lt;r*&gt;</td>

  </tr>

 </tbody>

</table>

<strong>Deliverables: </strong>Provide a single plot with the learning curves for these four runs. The run with both reward-to-go and the baseline should achieve an average score close to 200.

<strong>9      Bonus!</strong>

Choose any (or all) of the following:

<ul>

 <li>A serious bottleneck in the learning, for more complex environments, is the sample collection time. In infrastructure/rltrainer.py, we only collect trajectories in a single thread, but this process can be fully parallelized across threads to get a useful speedup. Implement the parallelization and report on the difference in training time.</li>

 <li>Implement GAE-<em>λ </em>for advantage estimation.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Run experiments in a MuJoCo gym environment to explore whether this speeds up training. (Walker2d-v1 may be good for this.)</li>

 <li>In PG, we collect a batch of data, estimate a single gradient, and then discard the data and move on. Can we potentially accelerate PG by taking multiple gradient descent steps with the same batch of data? Explore this option and report on your results. Set up a fair comparison between single-step PG and multi-step PG on at least one MuJoCo gym environment.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://arxiv.org/abs/1506.02438">https://arxiv.org/abs/1506.02438</a>