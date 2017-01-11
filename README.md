##Guide

####From stackoverflow

<http://stats.stackexchange.com/questions/36247/how-to-get-started-with-neural-networks>

Neural networks have been around for a while, and they've changed dramatically over the years. If you only poke around on the web, you might end up with the impression that "neural network" means multi-layer feedforward network trained with back-propagation. Or, you might come across any of the dozens of rarely used, bizarrely named models and conclude that neural networks are more of a zoo than a research project. Or that they're a novelty. Or...

If you want a clear explanation, I'd listen to [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton). He has been around forever and (therefore?) does a great job weaving all the disparate models he's worked on into one cohesive, intuitive (and sometimes theoretical) historical narrative. On his homepage, there are links to Google Tech Talks and Videolectures.net lectures he has done (on [RBMs](https://www.youtube.com/watch?v=AyzOUbkUf3M) and [Deep Learning](http://videolectures.net/jul09_hinton_deeplearn/), among others). 

From the way I see it, here's a historical and pedagogical road map to understanding neural networks, from their inception to the state-of-the-art:

- [Perceptrons](http://en.wikipedia.org/wiki/Perceptron)
    - Easy to understand
    - Serverely limited
- Multi-layer, trained by back-propogation
    - Many [resources](https://en.wikiversity.org/wiki/Learning_and_neural_networks) to learn these
    - Don't generally do as well as SVMs
- [Boltzmann machines](http://en.wikipedia.org/wiki/Boltzmann_machine)
    - Interesting way of thinking about the stability of a recurrent network in terms of "energy"
    - Look at [Hopfield networks](http://www.comp.leeds.ac.uk/ai23/reading/Hopfield.pdf) if you want an [easy to understand](http://lcn.epfl.ch/tutorial/english/hopfield/html/index.html) (but not very practical) example of recurrent networks with "energy".
    - Theoretically interesting, useless in practice (training about the same speed as continental drift)
- Restricted Boltzmann Machines
    - Useful!
    - Build off of the theory of Boltzmann machines
    - Some good [introductions](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/) on the web
- Deep Belief Networks
    - So far as I can tell, this is a class of multi-layer RBMs for doing semi-supervised learning.
    - Some [resources](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)




