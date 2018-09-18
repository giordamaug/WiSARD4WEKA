# WiSARD4WEKA
A supervised classification model for WEKA based on the WiSARD weightless neural network
for the Weka machine learning toolkit.

WiSARD was originally conceived as a pattern recognition device mainly focusing on image processing domain.
With ad hoc data transformation, WiSARD can also be used successfully as multiclass classifier in machine learning domain.

The WiSARD is a RAM-based neuron network working as an <i>n</i>-tuple classifier.
A WiSARD is formed by as many discriminators as the number of classes it has to discriminate between. 
Each discriminator consists of a set of <i>N</i> RAMs that, during the training phase, l
earn the occurrences of <i>n</i>-tuples extracted from the input binary vector (the <i>retina</i>).

In the WiSARD model, <i>n</i>-tuples selected from the input binary vector are regarded as the “features” of the input pattern to be recognized. It has been demostrated in literature [14] that the randomness of feature extraction makes WiSARD more sensitive to detect global features than an ordered map which makes a single layer system sensitive to detect local features.

More information and details about the WiSARD neural network model can be found in Aleksander and Morton's book [Introduction to neural computing](https://books.google.co.uk/books/about/An_introduction_to_neural_computing.html?id=H4dQAAAAMAAJ&redir_esc=y&hl=it).

The WiSARD4WEKA package implements a multi-class classification method based on the WiSARD weightless neural model
for the Weka machine learning toolkit. A data-preprocessing filter allows to exploit WiSARD neural model 
training/classification capabilities on multi-attribute numeric data making WiSARD overcome the restriction to
binary pattern recognition.

For more information on the WiSARD classifier implemented in the WiSARD4WEKA package, see:

<pre>
Massimo De Gregorio and Maurizio Giordano (2018). 
<i>An experimental evaluation of weightless neural networks for 
multi-class classification</i>.
Journal of Applied Soft Computing. Vol.72. pp. 338-354<br>
</pre>

If you use this software, please cite it as:

<pre>
&#64;article{DEGREGORIO2018338,
 title = "An experimental evaluation of weightless neural networks for multi-class classification",
 journal = "Applied Soft Computing",
 volume = "72",
 pages = "338 - 354",
 year = "2018",
 issn = "1568-4946",
 doi = "https://doi.org/10.1016/j.asoc.2018.07.052",
 url = "http://www.sciencedirect.com/science/article/pii/S156849461830440X",
 author = "Massimo De Gregorio and Maurizio Giordano",
 keywords = "Weightless neural network, WiSARD, Machine learning"
}
</pre>

# Building

This repository includes all sources, documentation and libraries to build the WiSARD4WEKA
package and jars. You can build the package jars by using the Apache <code>ant</code> utility:

<pre>
ant -f build_package.xml -Dpackage=WiSARD exejar
</pre>

Or you can build locally the package to be imported in your Weka toolkit by mean of the command:

<pre>
ant -f build_package.xml -Dpackage=WiSARD  make_package
</pre>

# Installation

This repository includes a pre-build package (in zip format) of WiSARD4WEKA that you can 
install from the the PackageManager of your Weka distribution:

<pre>
$ java -cp <your-path-to-weka.jar> weka.core.WekaPackageManager -install-package https://github.com/giordamaug/WiSARD4WEKA/releases/download/v.1.0.1/WiSARD.zip
</pre>

# Use

After package installation, you can run the WiSARD classifier from the Weka GUI:

![image](https://github.com/giordamaug/WiSARD4WEKA/blob/master/doc/wisard4weka1.png)

Once you select the WiSARD classifier, you can change its parameters:

![image](https://github.com/giordamaug/WiSARD4WEKA/blob/master/doc/wisard4weka2.png)

and then run the classifier on your dataset:

![image](https://github.com/giordamaug/WiSARD4WEKA/blob/master/doc/wisard4weka3.png)

# Experiments

Some measure of F-measure of WiSARD of classification on weka datasets (66\% split) in comparison with other methods:
(NOTE: all methods run in default configuration of paramters)

| method | pima-diabetes  | Glass | ionosphere | iris | labor-neg-data | soybean | segment | supermarket | vote | weather
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| WiSARD | 0.825          | 0.564 | **0.914**      |  1.000 |     0.889      |  1.000   |  0.986 | 0.779 | 0.989 | 1.0 |
| SMO    | **0.852**          | 0.275 | 0.749       | 1.000 |     0.889      |  1.000   |  1.000 | 0.779 | 0.972 | 0.75 |
| MLP    | 0.841          | 0.612 | 0.740       | 1.000 |     **0.923**      |  1.000   |  0.971 | 0.779 | 0.983 | 0.571 |
| j48    | 0.824          | **0.667** | 0.729       | 1.000 |     0.769      |  0.696   | 0.986 | 0.779 | 0.972 | 0.571 |
