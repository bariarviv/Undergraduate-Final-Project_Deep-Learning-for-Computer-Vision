# Developing a GAN for Generating MNIST Handwritten Digits

Generative Adversarial Network, or GAN, is an architecture for training generative models, such as deep convolutional neural networks for generating images. A GAN involves two deep learning networks pitted against each other in an adversarial relationship. One network is a generator that produces forgeries of images, and the other is a discriminator that attempts to distinguish the generator’s fakes from the real thing. The generator is tasked with receiving a random noise input and turning this into a fake image. The discriminator, a binary classifier of real versus fake images. Over several rounds of training, the generator becomes better at producing more convincing forgeries, and so too the discriminator improves its capacity for detecting the fakes. As training continues, the two models battle it out, trying to outdo one another, and, in so doing, both models become more and more specialized at their respective tasks. Eventually, this adversarial interplay can culminate in the generator producing fakes that are convincing not only to the discriminator network but also to the human eye.

<p align="center">
  <img src="images\1.png" width="500" height="250">
</p>

Training a GAN consists of two opposing processes:
* **Discriminator training:** in this process the generator produces fake images, that is, it performs inference only, while the discriminator learns to tell the fake images from real ones.
* **Generator training:** in this process the discriminator judges fake images produced by the generator. Here, it is the discriminator that performs inference only, whereas it’s the generator that uses this information to learn, in this case, to learn how to better fool the discriminator into classifying fake images as real ones.

<p align="center">
  <img src="images\2.png" width="450" height="200">
  <img src="images\3.png" width="450" height="200">
</p>

Thus, in each of these two processes, one of the models creates its output (either a fake image or a prediction of whether the image is fake) but is not trained, and the other model uses that output to learn to perform its task better.
At the onset of GAN training, the generator has no idea yet what it’s supposed to be making, so, being fed random noise as inputs, the generator produces images of random noise as outputs. These poor-quality fakes contrast starkly with the real images, which contain combinations of features that blend to form actual images, and therefore the discriminator initially has no trouble at all learning to distinguish real from fake. As the generator trains, however, it gradually learns how to replicate some of the structure of the real images. Eventually, the generator becomes crafty enough to fool the discriminator, and thus in turn the discriminator learns more complex and nuanced features from the real images such that outwitting the discriminator becomes trickier. Back and forth, alternating between generator training and discriminator training in this way, the generator learns to forge ever-more-convincing images. At some point, the two adversarial models arrive at a stalemate: they reach the limits of their architectures and learning stalls on both sides.

The GAN model was applied for the MNIST dataset. For every 10 epochs, the percentage of accuracy of the discriminator's classification was printed, and in addition, for the generator, a plot of the samples generated was presented.

## Requirements
~~~bash
pip install matplotlib
pip install tensorflow
pip install Keras
pip install numpy
~~~

## Results

<table align="center">
  <tr>
    <th>After 10 epochs:</th>
    <th>After 20 epochs:</th>
  </tr>
  <tr>
    <td><img src="results\After 10 epochs.png"></td>
    <td><img src="results\After 20 epochs.png"></td>
  </tr>
  <tr>
    <th>After 30 epochs:</th>
    <th>After 40 epochs:</th>
  </tr>
  <tr>
    <td><img src="results\After 30 epochs.png"></td>
    <td><img src="results\After 40 epochs.png"></td>
  </tr>
  <tr>
    <th>After 50 epochs:</th>
    <th>After 60 epochs:</th>
  </tr>
  <tr>
    <td><img src="results\After 50 epochs.png"></td>
    <td><img src="results\After 60 epochs.png"></td>
  </tr>
  <tr>
    <th>After 70 epochs:</th>
    <th>After 80 epochs:</th>
  </tr>
  <tr>
    <td><img src="results\After 70 epochs.png"></td>
    <td><img src="results\After 80 epochs.png"></td>
  </tr>
  <tr>
    <th>After 90 epochs:</th>
    <th>After 100 epochs:</th>
  </tr>
  <tr>
    <td><img src="results\After 90 epochs.png"></td>
    <td><img src="results\After 100 epochs.png"></td>
  </tr>
</table>
