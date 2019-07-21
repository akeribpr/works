How to use:
First - extract datas.rar into C:\ (C:\datas)
Open projects : Rewards/DQL - reinforcement/Actions and run them (every project is individual) - if now working on the first time try to run second time. (in the first time the projects are only resizing images).

Rewards

Basiclly , I took  an Alexnet network with pre trained weights, deleted the last layers of the fully connected.
Now every image that going into the network become a features vector. Then I measured the the distance betwean every frame image (features vector) 
to the last features vecror which is the last frame (the target frame).
The output is the graph of thease distances of every frame - the graph is seasonality because the video of the frames going Back and forth.

DQL - reinforcement

DQL-example file is the example for a reinforcement learning of cart pole problem.
I used that file for mainDQL file, which using a reinforcement learning to learn the best policy
to find the best way to reach the goal state of the probe.
First we run randomley with two actions (left and right) with initial indexes of frames to start from.
Then we store the transition (state, next state , action and reward) in order to improve the policy.
The output is the actions done for every iteration, the count of them for every episode, and the initialfrome index.
If we run the algorithm multiple times, we can see that the graph of the actions counter is converged.
Alexnet file here is calculation the reward by the measure of cosine similarity with the alexnet network.

Actions

First I divided two groups of images into two different files (one for frames which condidered to right moves and for left moves)
Then I resized and croped them.
Then I manipulate  a tranfare learning of Alexnet like I did before to get features vectors of the frames.
Then I built a network which get an input two features vectors (of the two frames that we consider as a move) and the output of the network
is two neurons. Then the new network learned the moves. 
In the learning process I determined that the right move got the weights [0,100] and left move [100,0]. 
