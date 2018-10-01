/*
 * ************************************************************************
 * This file is part of the WiSARD4WEKA distribution 
 * (https://github.com/WiSARD4WEKA).
 * Copyright (c) 2018 Maurizio Giordano.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * *************************************************************************/

package weka.classifiers.functions;

import weka.classifiers.functions.wisard.*;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Collections;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * <!-- globalinfo-start --> 
 * <b>Description:</b>
 * Implements a multi-class classification method based 
 * on the WiSARD weightless neural model.
 * A data-preprocessing filter allows to exploit WiSARD neural model 
 * training/classification capabilities on multi-attribute numeric data, 
 * whereas  WiSARD's  was historically restricted to binary pattern 
 * recognition. For more information on WiSARD classifier, see<p>
 *
 * Massimo De Gregorio and Maurizio Giordano (2018).<br> 
 * <i>An experimental evaluation of weightless neural networks for 
 * multi-class classification</i>.<br> 
 * Journal of Applied Soft Computing. Vol.72. pp. 338-354<p>
 *
 *<!-- globalinfo-end -->
 *
 *<!-- technical-bibtex-start --> <b>BibTeX:</b>
 * <pre>
 * &#64;article{DEGREGORIO2018338,
 * title = "An experimental evaluation of weightless neural networks for multi-class classification",
 * journal = "Applied Soft Computing",
 * volume = "72",
 * pages = "338 - 354",
 * year = "2018",
 * issn = "1568-4946",
 * doi = "https://doi.org/10.1016/j.asoc.2018.07.052",
 * url = "http://www.sciencedirect.com/science/article/pii/S156849461830440X",
 * author = "Massimo De Gregorio and Maurizio Giordano",
 * keywords = "Weightless neural network, WiSARD, Machine learning"
 * }
 * </pre>
 *<!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> <b>Valid options are:</b>
 * 
 * <pre>
 * -B &lt;bitno&gt;
 * The bit resolution. The value is an integer between 1 and 32.
 * </pre>
 * 
 * <pre>
 * -S &lt;ticno&gt;
 * The scaling range. The value is an integer in the range 1-8192.
 * </pre>
 * 
 * <pre>
 * -M &lt;LINEAR | RANDOM&gt;
 * * The mapping type. The options are LINEAR or RANDOM (with seed).
 * </pre>
 * 
 * <pre>
 * -m &lt;seed&gt;
 * Seed for random mapping. Can be -1 (no seed) of a valid noninteger seed.
 * </pre>
 * 
 *
 * <!-- options-end -->
 * 
 * @author Maurizio Giordano (maurizio.giordano@cnr.it)
 * @version $Revision: 1.0.1 $
 */
//<pre>
// -F
// Ture to enable bleaching algorithm (tie resolution). False for disabling it.
// </pre>
// 
// <pre>
// -s &lt;step&gt;
// Set bleaching step [nonegative real]
// </pre>
// 
// <pre>
// -c &lt;confidence&gt;
// Set bleaching confidence [real in 0,1]
// </pre>

public class WiSARD extends AbstractClassifier
implements OptionHandler, TechnicalInformationHandler {

	/** The training instances used for classification. */
	private Instances origInstances;
	private Instances trainingInstances;
	/** Filter for converting nominal attributes to binary ones */
	protected NominalToBinary m_NominalToBinary = null;
	/** Filter for replacing missing values */
	protected ReplaceMissingValues m_ReplaceMissingValues = null;
	/** The number of attributes. */
	protected int cardinality;
	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;
	/** Array of discriminators. */
	protected Discriminator[] darray;
	/** Number of classes. */
	private int m_NClasses;                       
	/** Class names list. */
	private String[] m_Classes;;                  
	/** attribute of the class. */
	private Attribute m_ClassAttr;                
	/** Number of features (minus the class). */
	private int m_NFeatures;                      
	/** Index of class in the attribute array. */
	private int m_cIdx;                           
	/** Mins of numeric data (discretization). */
	private double[] mins;                        
	/** Maxes of numeric data (discretization). */
	private double[] maxs;
	/** Ranges of numeric data (discretization). */
	private double[] ranges;
	/** True if attributes are all numeric. */
	private boolean m_onlyNumeric = true;

	/** Mapping random seed. */
	private long m_Seed = -1;                        
	/** Mapping type. */
	private mapType m_MapType = mapType.RANDOM;      
	/** Data discretization resolution. */
	private int m_TicNo = 256;                        
	/** Neuron bit resolution. */
	private int m_BitNo = 8;                         
	/** Bleaching stepping parameter. */
	private double m_BleachStep = 1.0;               
	/** Bleaching confidence parameter. */ 
	private double m_BleachConfidence = 0.01;        
	/** Bleaching enabling flag. */
	private boolean m_BleachFlag = false;            


	/**
	 * @return mapType [LINEAR | RANDOM].
	 */
	public mapType getMapType() {
		return m_MapType;
	}

	/**
	 * @param maptype is LINEAR if mapping is linear, otherwise it is RANDOM.
	 */
	public void setMapType(mapType maptype) {
		m_MapType = maptype;
	}

	/**
	 * @return Tooltip text describing the mapType option
	 */
	public String mapTypeTipText() {
		return "Set this to change the mapping type to either linear or random (with seed) ";
	}

	/**
	 * @return scaling range (number of tics).
	 */
	public int getTicNo() {
		return m_TicNo;
	}

	/**
	 * @param notics is the resolution in datum discretization.
	 */
	public void setTicNo(int notics) {
		m_TicNo = notics;
	}

	/**
	 * @return Tooltip text describing the ticNo option
	 */
	public String ticNoTipText() {
		return "Set the range of discrete values for datum";
	}

	/**
	 * @return neuron resolution (number of bits).
	 */
	public int getBitNo() {
		return m_BitNo;
	}

	/**
	 * @param bitno is the neuron resolution in number of bits.
	 */
	public void setBitNo(int bitno) {
		m_BitNo = bitno;
	}

	/**
	 * @return Tooltip text describing the bitNo option
	 */
	public String bitNoTipText() {
		return "Set the bit resolution of neurons";
	}

	/**
	 * @return bleaching enabling flag.
	 */
	/** public boolean getBleachFlag() {
		return m_BleachFlag;
	} */

	/**
	 * @param flag is True if bleaching is enblaed, False for disabling it.
	 */
	/** public void setBleachFlag(boolean flag) {
		this.m_BleachFlag = flag;
	} */

	/**
	 * @return Tooltip text describing the bleachFlag option
	 */
	/** public String bleachFlagTipText() {
		return "Enable the bleaching algorithm";
	} */

	/**
	 * @return bleaching step parameter.
	 */
	/** public double getBleachStep() {
		return m_BleachStep;
	} */

	/**
	 * @param step is the bleaching steping parameter.
	 */
	/** public void setBleachStep(double step) {
		this.m_BleachStep = step;
	} */

	/**
	 * @return Tooltip text describing the bleachFlag option
	 */
	/** public String bleachStepTipText() {
		return "Set the bleaching stepping";
	} */

	/**
	 * @return bleaching confidence parameter.
	 */
	/** public double getBleachConfidence() {
		return m_BleachConfidence;
	} */

	/**
	 * @param confidence is the bleaching confidence parameter.
	 */
	/** public void setBleachConfidence(double confidence) {
		this.m_BleachConfidence = confidence;
	} */

	/**
	 * @return Tooltip text describing the bleachConfidence option
	 */
	/** public String bleachConfidenceTipText() {
		return "Set the bleaching confidence";
	} */

	/**
	 * @return random mapping seed.
	 */
	public long getSeed() {
		return m_Seed;
	}

	/**
	 * @param seed is the seed for mapping randomization.
	 */
	public void setSeed(long seed) {
		this.m_Seed = seed;
	}

	/**
	 * @return Tooltip text describing the bleachConfidence option
	 */
	public String seedTipText() {
		return "Set the seed for mapping randomization";
	}

	/** 
	 * Setting capabilities of the Classifier
	 * @return capabilites of the classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		/**
		 * only numeric and nominal attributes are allowed
		 */
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		/**
		 *  only nominal classes are allowed
		 */
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		/**
		 * at least 1 instance has to be in the dataset
		 */
		//result.setMinimumNumberInstances(1);

		return result;
	}

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui.
	 */
	public String globalInfo() {
		return  "Implements De Gregorio & Giordano classifier algorithm based "
				+ "on weightless neural network.\n\n"
				+ "This implementation adds a data-preprocessing filter method to "
				+ "exploit WiSARD neural model training/classification capabilities "
				+ "(which are inherently restricted to binary pattern recognition) "
				+ "The resulting software is a classification method based on WiSARD "
				+ "neural network model. "
				+ "For more information on the WisardClassifier algorithm, see\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Generate technical information if classifier
	 * @return technical info
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(TechnicalInformation.Type.INBOOK);
		result.setValue(TechnicalInformation.Field.AUTHOR, "M. De Gregorio and M. Giordano");
		result.setValue(TechnicalInformation.Field.YEAR, "2018");
		result.setValue(TechnicalInformation.Field.TITLE, "An experimental evaluation of weightless neural networks for multi-class classification");
		result.setValue(TechnicalInformation.Field.JOURNAL, "Journal of Applied Soft Computing");
		result.setValue(TechnicalInformation.Field.EDITOR, "Elsevier");
		result.setValue(TechnicalInformation.Field.URL, "http://www.sciencedirect.com/science/article/pii/S156849461830440X");
		return result;
	}


	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @exception Exception if the classifier has not been generated 
	 * successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		/**
		 *  test data against capabilities
		 */
		//System.out.println(String.format("Training samples %d",instances.numInstances()));
		getCapabilities().testWithFail(instances);

		origInstances = new Instances(instances);
		/**
		 *  remove instances with missing class value,
		 *  but don't modify original data (use a copy)
		 */
		trainingInstances = new Instances(instances);
		trainingInstances.deleteWithMissingClass();

		/** 
		 * replace missing values
		 */
		trainingInstances = new Instances(trainingInstances);
		m_ReplaceMissingValues = new ReplaceMissingValues();
		m_ReplaceMissingValues.setInputFormat(trainingInstances);
		trainingInstances = Filter.useFilter(trainingInstances, m_ReplaceMissingValues);

		/** 
		 * convert nominal attributes (if required)
		 */
		m_onlyNumeric = true;
		for (int i = 0; i < trainingInstances.numAttributes(); i++) {
			if (i != trainingInstances.classIndex()) {
				if (!trainingInstances.attribute(i).isNumeric()) {
					m_onlyNumeric = false;
					break;
				}
			}
		}

		if (!m_onlyNumeric) {
			trainingInstances = new Instances(trainingInstances);
			m_NominalToBinary = new NominalToBinary();
			m_NominalToBinary.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, m_NominalToBinary);
		}
		cardinality = trainingInstances.numAttributes();

		/** 
		 * check where is the class attribute
		 */
		m_cIdx = trainingInstances.classIndex();
		m_ClassAttr = trainingInstances.attribute(m_cIdx);
		m_NClasses = trainingInstances.numClasses();
		m_NFeatures = trainingInstances.numAttributes()-1;
		m_Classes = new String[m_NClasses];

		for (int i=0; i < m_ClassAttr.numValues(); i++) {
			m_Classes[i] = m_ClassAttr.value(i);
		}

		/**
		 *  check if class is nominal
		 */
		if (trainingInstances.classAttribute().type() == Attribute.NUMERIC) {
			throw new Exception("Numeric class not supported in WisardClassifier!");
		}

		/**
		 *  initialize in,max and range arrays
		 */
		this.ranges = new double[m_NFeatures];
		this.mins = new double[m_NFeatures];
		this.maxs = new double[m_NFeatures];
		for (int i=0; i < m_NFeatures; i++) {
			this.mins[i] = Double.MAX_VALUE;
			this.maxs[i] = Double.MIN_VALUE;
		}

		/** 
		 * find min,max and range of values for each attribute
		 */
		for (Instance ist : trainingInstances) {
			int idx=0;
			double[] data = ist.toDoubleArray();
			for (int i=0; i <= m_NFeatures; i++) {
				if (i == m_cIdx) continue;
				if (data[i] > this.maxs[idx]) this.maxs[idx] = data[i];
				if (data[i] < this.mins[idx]) this.mins[idx] = data[i];
				idx += 1;
			}
		}
		for (int i=0; i < m_NFeatures; i++) {
			this.ranges[i] = this.maxs[i] - this.mins[i];
		}

		/**
		 *  check input parameters
		 */
		if (m_BitNo < 1 || m_BitNo > 32) {
			throw new IllegalArgumentException(
					"Bit resolution ranges in [1..32]!");
		}
		if (m_TicNo < 1 || m_TicNo > 8192) {
			throw new IllegalArgumentException(
					"Scaling range must be in [1..8192]!");
		}
		if (m_Seed < -1) {
			throw new IllegalArgumentException(
					"Mapping seed can be -1 (no seed) or a nonnegative (valid) seed !");
		}
		/** 
		if (m_BleachStep < 0) {
			throw new IllegalArgumentException(
					"Bleaching step must be non negative !");
		}
		if (m_BleachConfidence <= 0 || m_BleachConfidence >= 1) {
			throw new IllegalArgumentException(
					"Bleaching confidence must be in range 0,1 !");
		} **/

		/**
		 *  create discriminator array
		 */
		this.darray = new Discriminator[m_NClasses];
		for (int c=0; c < m_NClasses; c++) {
			this.darray[c] = new Discriminator(m_BitNo,m_TicNo * m_NFeatures,m_Classes[c],getMapType(), getSeed());
		}

		/**
		 * train the classifier
		 */
		for (Instance ist : trainingInstances) {
			updateClassifier(ist);
		}
	}

	/**
	 * Updates the classifier with the given instance.
	 *
	 * @param instance the new training instance to include in the model 
	 * @exception Exception if the instance could not be incorporated in
	 * the model.
	 */
	public void updateClassifier(Instance instance) throws Exception {
		// if datum has no classs... do nothing
		if (instance.classIsMissing()) return;

		/**
		 *  copy only attributes (not the class) into the wisard input (mdata).
		 */
		double[] data = instance.toDoubleArray();
		double[] mdata = new double[m_NFeatures];
		int idx = 0;
		for (int i=0; i <= m_NFeatures; i++) {
			if (i == m_cIdx) continue;
			mdata[idx] = data[i];
			idx += 1;
		}
		/**
		 * train the wisard disciminators
		 */
		this.darray[(int) data[instance.classIndex()]].trainHisto(mdata,this.ranges,this.mins,this.m_TicNo,this.m_NFeatures);
	}

	/**
	 * Classification routine.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception if there is a problem generating the prediction
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (!origInstances.equalHeaders(instance.dataset())) throw new Exception(
				"Incompatible instance types\n" + trainingInstances.equalHeadersMsg(instance.dataset()));

		if (trainingInstances.numInstances() == 0) {
			throw new Exception("No training instances!");
		}
		if (trainingInstances.numClasses() == 1) {
			if (getDebug()) System.out.println("Training data have only one class");
			// 100 percent likelihood of belonging to the one class
			return new double[]{1};
		}
		// trasform nominal to binary attributes (if required)
		if (!m_onlyNumeric) {
			m_NominalToBinary.input(instance);
			instance = m_NominalToBinary.output();
		}
		double[] dist = new double[m_NClasses];
		int pSum = 0;

		/**
		 *  copy only attributes (not the class) into the wisard input (mdata).
		 */
		double[] data = instance.toDoubleArray();
		double[] mdata = new double[m_NFeatures];
		int idx = 0;
		for (int i=0; i <= m_NFeatures; i++) {
			if (i == m_cIdx) continue;
			mdata[idx] = data[i];
			idx += 1;
		}
		/**
		 *  get wisard predictions
		 */
		if (this.m_BleachFlag)  {    // Bleaching is enabled
			double b = this.m_BleachStep;
			double confidence = 0.0;
			int n_rams = this.darray[0].getN_ram();
			for (int c=0; c < m_NClasses; c++) {      // calculate responses of each discriminator
				this.darray[c].responseHisto(mdata,this.ranges,this.mins,this.m_TicNo,m_NFeatures);
			}
			int[] result_partial = new int[m_NClasses];
			while (confidence < this.m_BleachConfidence) {
				pSum = 0;
				for (int c=0; c < m_NClasses; c++) {   
					result_partial[c] = 0;
					for (int neuron = 0; neuron < n_rams; neuron++) {
						if (this.darray[c].getResponse()[neuron] > b)
							result_partial[c]++;
					}
					pSum += result_partial[c];
				}
				confidence = calc_confidence(result_partial);
				if (confidence < 0)
					throw new Exception("Something wrong in Bleaching algorithm!");
				b += 1.0;
				if (pSum == 0) {
					pSum = 0;
					for (int c=0; c < m_NClasses; c++) {   
						result_partial[c] = 0;
						for (int neuron = 0; neuron < n_rams; neuron++) {
							if (this.darray[c].getResponse()[neuron] >= 1.0)
								result_partial[c]++;
						}
						pSum += result_partial[c];
					}
					break;
				}
				for (int c=0; c < m_NClasses; c++) 
					System.out.println(String.format("%f ",result_partial[c]));
				System.out.println("\n");
			}
			if (pSum == 0) {
				for (int c=0; c < m_NClasses; c++) {
					dist[c] = 0.0;
					for (int neuron = 0; neuron < n_rams; neuron++)
						dist[c] += this.darray[c].getResponse()[neuron];
					dist[c] = dist[c] / (double) n_rams;
				}
			} else
				for (int c=0; c < m_NClasses; c++)  
					dist[c] = result_partial[c] / (double) pSum;
		}
		else                        // No Bleaching!
			for (int c=0; c < m_NClasses; c++) {
				dist[c] = this.darray[c].classifyHisto(mdata,this.ranges,this.mins,this.m_TicNo,m_NFeatures);
				// wisard prediction are normalize to get a distribution.
			}
		return dist;

	}

	/**
	 * Classifies the given test instance. The instance has to belong to a
	 * dataset when it's being classified. Note that a classifier MUST
	 * implement either this or distributionForInstance().
	 *
	 * @param instance the instance to be classified
	 * @return the predicted most likely class for the instance or 
	 * Instance.missingValue() if no prediction is made
	 * @exception Exception if an error occurred during the prediction
	 */
	public double classifyInstance(Instance instance) throws Exception {

		double[] dist = distributionForInstance(instance);
		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}
		switch (instance.classAttribute().type()) {
		case Attribute.NOMINAL:
			double max = 0;
			int maxIndex = 0;

			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}
			if (max > 0) {
				return maxIndex;
			} else {
				// throw new Exception("Predicted missing value.");
				return 0;
			}
		case Attribute.NUMERIC:
			throw new Exception("Numeric class not supported in WisardClassifier!");
		default:
			// throw new Exception("Predicted missing value.");
			return 0;
		}
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Enumeration   	en;
		Vector result;

		result = new Vector();

		result.add(new Option("\tSet bit resolution\n","B", 16,"-B <bitno>"));
		result.addElement(new Option("\tSet scaling range\n","S", 128,"-S <ticno>"));
		result.addElement(new Option("\tSet mapping type [linear, random]\n", "M <maptype>", mapType.RANDOM.ordinal(),"-M"));
		result.addElement(new Option("\tSet mapping seed\n", "m", -1,"-m <seed>"));
		//result.addElement(new Option("\tEnable Bleaching\n", "F", 0,"-F"));
		//result.addElement(new Option("\tSet Bleaching step\n", "s", 1,"-s <step>"));
		//result.addElement(new Option("\tSet Bleaching confidence\n", "c", 1,"-c <confidence>"));

		en = super.listOptions();
		while (en.hasMoreElements())
			result.addElement(en.nextElement());

		return result.elements();
	}

	/**
	 * Gets the options of super.
	 *
	 * @return Vector of all Options given in parent object(s).
	 */
	private Vector<Option> getOptionsOfSuper() {
		Vector<Option> v = new Vector<Option>();
		// super will always return Enumeration<Option>
		Enumeration<Option> e = super.listOptions();
		while (e.hasMoreElements()) {
			Option option = e.nextElement();
			v.add(option);
		}
		return v;
	}

	/**
	 * <!-- options-start --> * Valid options are: <p> * *
	 * 
	 * <pre>
	 * -B &lt;bitno&gt;
	 * * The bit resolution. The value is an integer between 1 and 32.
	 * </pre>
	 * 
	 * <pre>
	 * -S &lt;ticno&gt;
	 * * The scaling range. The value is an integer in the range 1-8192.
	 * </pre>
	 * 
	 * <pre>
	 * -M &lt;LINEAR | RANDOM&gt;
	 * * The mapping type. The options are LINEAR or RANDOM (with seed).
	 * </pre>
	 * 
	 * <pre>
	 * -m &lt;seed&gt;
	 * * Seed for random mapping. Can be -1 (no seed) of a valid noninteger seed.
	 * </pre>
	 * 
	 * * <!-- options-end -->
	 * @param options {@inheritDoc}
	 */
	// <pre>
	// -F
	// True to enable bleaching algorithm (tie resolution). False for disabling it.
	// </pre>
	//  
	// <pre>
	//  -s &lt;step&gt;
	//  Set bleaching step [nonegative real]
	//  </pre>
	//  
	//  <pre>
	//  -c &lt;confidence&gt;
	//  Set bleaching confidence [real in 0,1]
	//  </pre>
	// 
	@Override
	public void setOptions(String[] options) {
		try {
			String optionString;
			// parse flag option
			/* if (Utils.getFlag('F', options)) setBleachFlag(true);
			else 
				setBleachFlag(false); */
			/**
			 *  parse integer options
			 */
			optionString = Utils.getOption('B', options);
			if (optionString.length() != 0) 
				setBitNo(Integer.parseInt(optionString));
			else
				setBitNo(8);
			optionString = Utils.getOption('S', options);
			if (optionString.length() != 0) 
				setTicNo(Integer.parseInt(optionString));    
			else
				setTicNo(256);
			optionString = Utils.getOption('m', options);
			if (optionString.length() != 0) 
				setSeed(Integer.parseInt(optionString));
			else
				setSeed(-1);
			/**
			 *  parse float options
			 */
			/* optionString = Utils.getOption('s', options);
			if (optionString.length() != 0) 
				setBleachStep(Double.parseDouble(optionString));
			else
				setBleachStep(1.0);
			optionString = Utils.getOption('c', options);
			if (optionString.length() != 0) 
				setBleachConfidence(Double.parseDouble(optionString));
			else
				setBleachConfidence(0.01); */
			/**
			 *  parse string (enum) options
			 */
			optionString = Utils.getOption('M', options);
			if (optionString.length() != 0) 
				setMapType(mapType.valueOf(optionString));
			else
				setMapType(mapType.RANDOM);

			Utils.checkForRemainingOptions(options);

		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	/**
	 * Get the options of the current setup.
	 * @return		the current options
	 */
	@Override
	public String[] getOptions() {
		int i;
		String[] options;
		Vector<String>    	result = new Vector<>();

		//if (getBleachFlag()) result.add("-F"); 
		result.add("-B"); result.add(String.format("%d",getBitNo()));
		result.add("-S"); result.add(String.format("%d",getTicNo()));
		result.add("-M"); result.add(String.format("%s",getMapType().name()));
		result.add("-m"); result.add(String.format("%d", getSeed()));
		//result.add("-s"); result.add(String.format(Locale.US,"%.3f",getBleachStep()));
		//result.add("-c"); result.add(String.format(Locale.US,"%.3f",getBleachConfidence()));

		options = super.getOptions();
		for (i = 0; i < options.length; i++)
			result.add(options[i]);

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Returns a description of the classifier.
	 *
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("WiSARD Classifier (2018 Maurizio Giordano)\n");
		sb.append("bits: ").append(getBitNo());
		sb.append("\n");
		sb.append("tics: ").append(getTicNo());
		sb.append("\n");
		if (trainingInstances != null) {
			sb.append("Training instances: ").append(trainingInstances.size());
			sb.append("\n");
		}
		return sb.toString();
	}




	/**
	 * Get revision number
	 * @return revision number
	 */
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return RevisionUtils.extract("$Revision: 2.0 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv the options
	 */
	public static void main(String[] argv) {
		runClassifier(new WiSARD(), argv);
	}

	/** 
	 * Bleaching confidence update function
	 * @param results
	 * @return confidence
	 */
	private double calc_confidence(int[] results) {
		// find the max (and the second max)
		int first, second, firstIndex=0;

		first = second = Integer.MIN_VALUE;
		for (int i = 1; i < results.length; i++) {
			if (results[i] > first) { /* If current element is greater than first then update both first and second */
				second = first;
				first = results[i];
				firstIndex = i;
			} else if (results[i] > second && results[i] != first)   /* If element in between first and second then update second  */
				second = results[i];
		}
		// if max is zero, confidence is zero
		if (first == 0) return 0;
		// check if there are 2 or more max
		for (int i = 0; i < results.length; i++)
			if (results[i] == first && i != firstIndex)
				return 0;   // with two max confidence is zero

		if (second == Integer.MIN_VALUE)
			return -1;
		else
			return (1 - (float) second / (float) first);

	}

}














