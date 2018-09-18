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
Massimo De Gregorio and Maurizio Giordano (2018).<br> 
<i>An experimental evaluation of weightless neural networks for 
multi-class classification</i>.<br> 
Journal of Applied Soft Computing. Vol.72. pp. 338-354<br>
</pre>

If you use this software, please cite it as:

<code>
&#64;article{DEGREGORIO2018338,<br>
 title = "An experimental evaluation of weightless neural networks for multi-class classification",<br>
 journal = "Applied Soft Computing",<br>
 volume = "72",<br>
 pages = "338 - 354",<br>
 year = "2018",<br>
 issn = "1568-4946",<br>
 doi = "https://doi.org/10.1016/j.asoc.2018.07.052",<br>
 url = "http://www.sciencedirect.com/science/article/pii/S156849461830440X",<br>
 author = "Massimo De Gregorio and Maurizio Giordano",<br>
 keywords = "Weightless neural network, WiSARD, Machine learning"<br>
}
</code>

# Install

You can install WiSARD4WEKA from the the PackageManager of your Weka distribution:

<code>
$ java -cp <your-path-to-weka.jar> weka.core.WekaPackageManager -install-package https://github.com/giordamaug/WiSARD4WEKA/releases/download/v.1.0.1/WiSARD.zip
</code>
