# 用深度强化学习玩Atari

# Abstract

我们提出了第一个深度学习模型，用强化学习来直接地从高维感官输入中学习控制策略。该
model是一个卷积神经网络，使用Q-learning的变体进行训练，其输入是原始像素，其输出对未来reward进行估计的值函数。 我们将方法应用于Arcade Learning Environment的七款Atari 2600游戏
上，在不调整架构或学习算法的情况下。我们发现它在六款游戏中都超越了之前的所有方法的性能，并且成功在其中三个里超越了一个人类玩家。

# Introduction

Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning (RL). Most successful RL applications that operate on these domains have relied on hand-crafted features combined with linear value functions or policy representations. Clearly, the performance of such systems heavily relies on the quality of the feature representation.

从视觉和语言等高维感觉输入中直接学习控制agent，是强化学习（RL）的长期挑战之一。大多数成功的RL应用程序在这些域上运行的都依赖于人工构造的features，并结合线性值函数或策略表示。显然，这种系统的性能很大程度上依赖于特征表示的质量。

深度学习的最新进展使得从原始感觉数据中提取高级特征成为可能，从而导致计算机视觉[11,22,16]和语音识别[6,7]的突破。这些方法利用一系列神经网络架构，包括卷积网络，多层感知器，受限玻尔兹曼机器和递归神经网络，并利用有监督和无监督学习。似乎很自然地会问，类似的技术是否也可能对感觉数据的RL有益。

然而，强化学习从深度学习的角度提出了一些挑战。首先，迄今为止大多数成功的深度学习应用都需要大量的手工标记的训练数据。另一方面，RL算法必须能够从频繁稀疏，嘈杂和延迟的标量奖励信号中学习。与监督学习中的输入和目标之间的直接关联相比，行动和由此产生的奖励之间的延迟（可能是数千次步长）似乎特别令人生畏。另一个问题是大多数深度学习算法都假设数据样本是独立的，而在强化学习中则是假设
通常会遇到高度相关状态的序列。此外，在RL中，数据分布随着算法学习新行为而改变，这对于假定固定底层分布的深度学习方法可能是有问题的。

Recent advances in deep learning have made it possible to extract high-level features from raw sensory data, leading to breakthroughs in computer vision [11, 22, 16] and speech recognition [6, 7]. These methods utilise a range of neural network architectures, including convolutional networks, multilayer perceptrons, restricted Boltzmann machines and recurrent neural networks, and have exploited both supervised and unsupervised learning. It seems natural to ask whether similar techniques could also be beneficial for RL with sensory data. 

However reinforcement learning presents several challenges from a deep learning perspective. Firstly, most successful deep learning applications to date have required large amounts of handlabelled training data. RL algorithms, on the other hand, must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning. Another issue is that most deep learning algorithms assume the data samples to be independent, while in reinforcement learning one
typically encounters sequences of highly correlated states. Furthermore, in RL the data distribution changes as the algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed underlying distribution.