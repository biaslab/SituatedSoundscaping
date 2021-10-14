# A Bayesian Modeling Approach to Situated Design of Personalized Soundscaping Algorithms
*By Bart van Erp, Albert Podusenko, Tanya Ignatenko and Bert de Vries*
### Published in the special issue on AI, Machine Learning and Deep Learning in Signal Processing of the Applied Sciences journal (2021).
---
**Abstract**

Effective noise reduction and speech enhancement algorithms have great potential to enhance lives of hearing aid users by restoring speech intelligibility. An open problem in today’s commercial hearing aids is how to take into account users’ preferences, indicating which acoustic sources should be suppressed or enhanced, since they are not only user-specific but also depend on many situational factors. In this paper, we develop a fully probabilistic approach to “situated soundscaping”, which aims at enabling users to make on-the-spot (“situated”) decisions about the enhancement or suppression of individual acoustic sources. The approach rests on a compact generative probabilistic model for acoustic signals. In this framework, all signal processing tasks (source modeling, source separation and soundscaping) are framed as automatable probabilistic inference tasks. These tasks can be efficiently executed using message passing-based inference on factor graphs. Since all signal processing tasks are automatable, the approach supports fast future model design cycles in an effort to reach commercializable performance levels. The presented results show promising performance in terms of SNR, PESQ and STOI improvements in a situated setting.

---
This repository contains the experiments and derivations of the paper, available at [https://www.mdpi.com/2076-3417/11/20/9535](https://www.mdpi.com/2076-3417/11/20/9535).