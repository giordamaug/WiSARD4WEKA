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

package weka.classifiers.functions.wisard;

import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.Vector;
import weka.core.RevisionHandler;

/**
 * <!-- globalinfo-start --> 
 * <b>Description:</b>
 * Implements the WiSARD weightless neural model.
 * The Discriminator is an array of RAMs with 2^n storage locations.
 * Each RAM is stimulated in writing (training) and reading (classification) mode
 * by accessing one of its locations by means of an input integer address (between 0 and 2<SUP>n</SUP> - 1).
 * a <i>tuple</i> is an array of integer addresses for RAMs, with as many elements as the 
 * number of RAMs.<p>
 * Depending on the type of input data to learn/classify we need to encode the input into
 * a array of integer addresses for RAMs.<ul>
 * <li> if the input is a binary vector, then a partition of it into <i>n</i>-sized tuples is
 * obtained (linearly or randomly). Extracted tuples of bits are decoded into integers used to 
 * access RAMs' locations.</li>
 * <li> if the input is a vector of real data with size <i>m</i>, first, the vector values are scaled-up and
 * discretized to a value <i>v</i> in the range of integers: 0, <i>z</i>-1; each value is translated into the
 * <i>thermometer encoding</i>: a sequence of <i>v</i> bits set followed by <i>z-v</i> bits unset. 
 * The bits sequences are arranged one after the other to form a binary input vector with size <i>z &times; m</i>. 
 * Then, the encoding of the binary input vector is like in the previous case.
 * </li>
 * </ul>
 *  
 * For more information on WiSARD model, see<p>
 *
 * Massimo De Gregorio and Maurizio Giordano (2018).<br> 
 * <i>An experimental evaluation of weightless neural networks for 
 * multi-class classification</i>.<br> 
 * Journal of Applied Soft Computing. Vol.72. pp. 338-354<br>
 *
 *<!-- globalinfo-end -->
 **/

public class Discriminator implements Serializable, RevisionHandler {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/** number of rams for the discriminator */
	private int n_ram;          								
	/** number of location in RAMs */
	private long n_loc;          								
	/** number of bits (resolution) */
	private int n_bit;          								
	/** size of input binary image */
	private int size;           								
	/** train counter */
	private long tcounter;  									
	/** seed for mapping */
	private long seed;  									    
	/** the ram list of disciminator */
	private Ram[] rams;											
	/** the list of more frequent keys */
	private Wentry[] maxkeys;    								
	/** pointer to the retina (mapping) */
	private int[] map;           						        
	/** pointer to the inverse retina (mapping) */
	private int[] rmap;          						        
    /** pointer to mental image */ 
	private double[] mi;        							    
    /** max mental image value */
	private double maxmi;      									
    /** response list */
	private double[] response;       							
    /** the name of the discriminator */
	private String name;         								
    /** the mapping type */
	private mapType maptype;									
    
    
    long mypowers[] = new long[] { 1L, 2L, 4L, 8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L, 4096L, 8192L, 16384L, 32768L, 65536L, 131072L , 262144L, 524288L,
    	    1048576L, 2097152L, 4194304L, 8388608L, 16777216L, 33554432L, 67108864L, 134217728L, 268435456L, 536870912L, 1073741824L, 2147483648L,
    	    4294967296L, 8589934592L, 17179869184L, 34359738368L, 68719476736L, 137438953472L, 274877906944L, 549755813888L, 1099511627776L, 2199023255552L, 
    	    4398046511104L, 8796093022208L, 17592186044416L, 35184372088832L, 70368744177664L, 140737488355328L, 281474976710656L, 562949953421312L, 1125899906842624L, 
    	    2251799813685248L, 4503599627370496L, 9007199254740992L, 18014398509481984L, 36028797018963968L, 72057594037927936L, 144115188075855872L, 288230376151711744L, 
    	    576460752303423488L, 1152921504606846976L, 2305843009213693952L, 4611686018427387904L, 
    	    //9223372036854775808L
    	   };
    
    class mapPair{
    	 public int[] map1;
    	 public int[] map2;
    }
    
    static void shuffleArray(int[] ar, long seed)
    {
      // If running on Java 6 or older, use `new Random()` on RHS here
      Random rnd;
      if (seed >= 0)  rnd = new Random(seed);
      else  rnd = new Random();
      for (int i = ar.length - 1; i > 0; i--)
      {
        int index = rnd.nextInt(i + 1);
        // Simple swap
        int a = ar[index];
        ar[index] = ar[i];
        ar[i] = a;
      }
    }
    
    protected mapPair mapping() throws WisardException {
        int i;
        mapPair maps = new mapPair();
        maps.map1 = new int[this.size];
        maps.map2 = new int[this.size];

        int[] list = new int[this.size];
        int[] rlist = new int[this.size];
        for (i = 0; i < this.size; i++) {
            list[i] = i;
            rlist[i] = i;
        }      
        if (this.maptype == mapType.RANDOM) {
        	shuffleArray(list, this.seed);
        	for (i = 0; i < this.size; i++) rlist[list[i]] = i;
            maps.map1 = list;
            maps.map2 = rlist;
        } else if (this.maptype == mapType.LINEAR) {
            maps.map1 = list;
            maps.map2 = rlist;
        } else {
        	throw new WisardException( "received wrong mapping mode" );
        }
        return maps;        
    }

    protected void init() {
    	n_bit = 16;
    	n_loc = mypowers[n_bit];
    	tcounter = (long) 0;
    	maxmi = (double) 0;
    	name = "Anonym";
    	maptype = mapType.RANDOM;
    }

    protected void fininit() {
    	this.mi = new double[this.size];
    	int neuron;
    	if (this.size % n_bit == 0) {
    		this.n_ram = (int)(this.size / n_bit);
    	} else {
    		this.n_ram = (int)(this.size / n_bit) + 1;
    	}
    	this.rams = new Ram[this.n_ram];
    	this.response = new double[this.n_ram];
    	this.maxkeys = new Wentry[this.n_ram];
    	for (neuron = 0; neuron < this.n_ram; neuron++) {
    		this.rams[neuron] = new Ram();
    		this.maxkeys[neuron] = new Wentry((long) 0, -1.0);
    	}
    }

    protected void computeMaps() throws WisardException {
    	mapPair maps = mapping();
    	this.map = maps.map1;
    	this.rmap = maps.map2;
    }

    /**
     * Discriminator constructor
     * @param n_bit number of bits for keys of RAMs
     * @param size the input binary input size
     * @param name of discriminator
     * @param mt the mapping type (the options are LINEAR or RANDOM)
     * @param seed seed random mapping. Can be -1 (no seed) of a valid noninteger seed.
     * @throws WisardException error in construction
     */
    public Discriminator(int n_bit, int size, String name, mapType mt, long seed) throws WisardException {
        init(); 
        if (n_bit > 32) throw new WisardException("Up to 32 bit supported!");
        this.n_bit = n_bit;
        this.n_loc = mypowers[n_bit];
        this.size = size;
        this.name = name;
        this.maptype = mt;
        this.seed = seed;
        fininit();
        computeMaps();
    }

    /**
     * Discriminator constructor
     * @param size the input binary input size
     * @param name of discriminator
     * @param mt the mapping type (the options are LINEAR or RANDOM)
     * @throws WisardException error in construction
     */
    public Discriminator(int size, String name, mapType mt) throws WisardException {
        init();    	
        this.size = size;
        this.name = name;
        this.maptype = mt;
        fininit();
        computeMaps();
    }
    
    /**
     * Discriminator constructor
     * @param size the input binary input size
     * @param name of discriminator
     * @throws WisardException error in construction
     */
    public Discriminator(int size, String name) throws WisardException {
        init();    	
        this.size = size;
        this.name = name;
        fininit();
        computeMaps();
    }
    /**
     * Discriminator constructor
     * @param size the input binary input size
     * @throws WisardException error in construction
     */
    public Discriminator(int size) throws WisardException {
        init();    	
        this.size = size;
        fininit();
        computeMaps();
    }
    
    /**
     * Check if the input is a legal tuple for the discriminator.
     * @param tuple the input tuple of keys
     * @return true if the input is a legal tuple for the discriminator
     */
    public Boolean checkTuple(long[] tuple) {
        if (tuple.length != this.n_ram) {
        	return false;
        }
        for (int i=0; i < tuple.length; i++) {
			if (tuple[i] > (this.n_loc - (long)1)) {
				return false;
			}
		}
		return true;
    	
    }
    /**
     * Getter funtion for Memroy Image of discriminator.
     * Upon access, the memory image is updated and stored internally in the discriminator
     * @return memory max value of memory image 
     */
    double updateMI() {
        int neuron, i, offset=0, b;
        double maxvalue=0, value;
        for (i=0; i< this.size; i++) this.mi[i] = 0;
        for (neuron=0,offset=0;neuron<this.n_ram;neuron++,offset+=this.n_bit) {
            Iterator<Wentry> it = this.rams[neuron].wentries.iterator (); 
            while (it.hasNext ()) { 
                Wentry entry = (Wentry) it.next (); 
                for (b=0;b<this.n_bit;b++) {
                	if (((entry.key)>>(long)(this.n_bit - 1 - b) & 1) > 0) {
                        value = this.mi[this.map[(offset + b) % this.size]] += entry.value;
                        if (maxvalue < value) maxvalue = value;
                    }
                }
            } 
        }
        this.maxmi = maxvalue;
        return maxvalue;
    }
    
    /**
     * Train on the discriminator on one sample of data.
     * Before training, input data vector is scaled and discretized to a integer vector; 
     * then binarized using the 'thermometer' encoding. 
     * the resulting binary vector is transformed to a set of keys to access discriminators RAMs. 
     * 
     * @param data the input vector of real numbers
     * @param range the array of value intervals of each component of data
     * @param off the array of offsets (min values) of each component of data
     * @param z the range of integer values (from 0 to z-1) the datum may assume after discretization 
     * @param nattr the number of data vector components (attributes of sample data)
     */
    public void trainHisto(double data[], double range[], double off[], int z, int nattr) {
        int neuron;
        double retval;
        this.tcounter++;
        long address;
        int x, i, index, npixels=z * nattr, value;

        if (data.length != this.size) {
        	
        }
        for (neuron=0;neuron<this.n_ram;neuron++) {
            // compute neuron simulus
            address=(long)0;
            // decompose record data values into wisard input
            for (i=0;i<this.n_bit;i++) {
                x = this.map[(((neuron * this.n_bit) + i) % npixels)];
                index = x/z;
                value = (int) ((data[index] - off[index]) * z / range[index]);
                if ((x % z) < value) {
                    address |= mypowers[this.n_bit -1 - i];
                }
           }
            retval = this.rams[neuron].write(address);
            // update max key for neuron
            if (retval > this.maxkeys[neuron].value) {
                this.maxkeys[neuron].key = address;
                this.maxkeys[neuron].value = retval;
            }
        }
    }
    /**
     * Train on the discriminator on one input tuple of keys to access discriminators RAMs
     * 
     * @param tuple the input tuple of keys for RAMs
     * @throws WisardException error if tuple is not legal for the discriminator
     */
    public void train(long[] tuple) throws WisardException {
        int neuron;
        double retval;
		if (!this.checkTuple(tuple)) throw new WisardException("Wrong tuple size or value");
        this.tcounter++;
        for (neuron=0;neuron<this.n_ram;neuron++) {
            retval = this.rams[neuron].write(tuple[neuron]);
            // update max key for ram
            if (retval > this.maxkeys[neuron].value) {
            	this.maxkeys[neuron].key = tuple[neuron];
            	this.maxkeys[neuron].value = retval;
            }
        }
    }
    /**
     * Classify one sample of data by summing-then-averaging responses of discriminators.
     * (responses are not stored in discriminator objects).
     * Before classification, input data vector is scaled and discretized to a integer vector; 
     * then binarized using the 'thermometer' encoding. 
     * the resulting binary vector is transformed to a set of keys to access discriminators RAMs. 
     * 
     * @param data the input vector of real numbers
     * @param range the array of value intervals of each component of data
     * @param off the array of offsets (min values) of each component of data
     * @param z the range of integer values (from 0 to z-1) the datum may assume after discretization 
     * @param nattr the number of data vector components (attributes of sample data)
     * @return the classification response (a double in the range 0,1)
     */
    public double classifyHisto(double data[], double range[], double off[], int z, int nattr) {
        int neuron, sum=0;
        long address;
        int x, i, index, npixels=z * nattr, value;
        
        for (neuron=0;neuron<this.n_ram;neuron++) {
            // compute neuron simulus
            address=(long)0;
            // decompose record data values into wisard input
            for (i=0;i<this.n_bit;i++) {
                x = this.map[((neuron * this.n_bit) + i) % npixels];
                index = x/z;
                value = (int) ((data[index] - off[index]) * z / range[index]);
                if ((x % z) < value) {
                    address |= (long)mypowers[this.n_bit -1 - i];
                }
            }
            if (this.rams[neuron].read(address) > 0) {
                sum++;
            }
        }
        // store responses
        return (double)sum/(double)this.n_ram;
    }
    /** 
     * Classify on an input tuple of keys to access discriminators RAMs.
     * @param tuple the input tuple of keys for RAMs
     * @return the classification response (a double in the range 0,1)
     * @throws WisardException error if tuple is not legal for the discriminator
     */
    public double classify(long[] tuple) throws WisardException {
        int neuron, sum=0;
        
		if (!this.checkTuple(tuple)) throw new WisardException("Wrong tuple size or value");
        for (neuron=0;neuron<this.n_ram;neuron++) {
            if (this.rams[neuron].read(tuple[neuron]) > 0) {
                sum++;
            }
        }
        // store responses
        return (double)(sum/(double)this.n_ram);
    }
    /**
     * Computes (and stores) responses of each discriminator on one sample of data.
     * Before getting responses, input data vector is scaled and discretized to a integer vector; 
     * then binarized using the 'thermometer' encoding. 
     * the resulting binary vector is transformed to a set of keys to access discriminators RAMs. 
     * 
     * @param data the input vector of real numbers
     * @param range the array of value intervals of each component of data
     * @param off the array of offsets (min values) of each component of data
     * @param z the range of integer values (from 0 to z-1) the datum may assume after discretization 
     * @param nattr the number of data vector components (attributes of sample data)
     */
    public void responseHisto(double data[], double range[], double off[], int z, int nattr) {
        int neuron;
        long address;
        int x, i, index, npixels=z * nattr, value;
        
        for (neuron=0;neuron<this.n_ram;neuron++) {
            // compute neuron simulus
            address=(long)0;
            // decompose record data values into wisard input
            for (i=0;i<this.n_bit;i++) {
                x = this.map[((neuron * this.n_bit) + i) % npixels];
                index = x/z;
                value = (int) ((data[index] - off[index]) * z / range[index]);
                if ((x % z) < value) {
                    address |= (long)mypowers[this.n_bit -1 - i];
                }
            }
            this.response[neuron] = this.rams[neuron].read(address);
        }
    }
    /**
     * Computes (and stores) responses of each discriminator on an input tuple of keys to access discriminators RAMs.
     * @param tuple the input tuple of keys for RAMs
     * @throws WisardException error if tuple is not legal for the discriminator
     */
    public void response(long[] tuple) throws WisardException {
        int neuron;
        
		if (!this.checkTuple(tuple)) throw new WisardException("Wrong tuple size or value");
        for (neuron=0;neuron<this.n_ram;neuron++) {
            this.response[neuron] = this.rams[neuron].read(tuple[neuron]);
        }
    }
    /**
     * Print funtion for discriminator
     * @return the printout (string) of the discriminator info
     */
    public String toString() {
		int neuron, i;
		String str = "[";
		str += String.format("class: %s bits: %d, nram: %d, loc: %d\n", this.name, this.n_bit, this.n_ram, this.n_loc);
		str += "map: [";
		for (i=0; i < this.size; i++) 
			str += String.format("%d ", this.map[i]);
		str += "]\n";
		str += "rmap: [";
		for (i=0; i < this.size; i++) 
			str += String.format("%d ", this.rmap[i]);
		str += "]\n";
		for (neuron = 0; neuron < this.n_ram; neuron++) {
			str += this.rams[neuron].toString();
		}
        return str +  "]\n";
	}
    /**
     * Print memory inage of discriminator
     * @return the printout (string) of the memory image
     */
	public String MItoString() {
		String str = "[MI = \n";
		int i;
		System.out.println(this.mi.length);
		for (i=0; i < this.mi.length; i++) {
			str += String.format("%.0f ", this.mi[i]);
		}
        return str +  "]\n";
	}
    /**
     * Return memory image of discriminator
     * @return a vector of double representing the memory image
     */
	public double[] MItoVector() {
		mi = new double[this.mi.length];
		for (int i=0; i < this.mi.length; i++)
			mi[i] = this.mi[i];
		return mi;
	}
	
	/**
	 * Get revision number
	 * @return revision number
	 */
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**	
	 * Getter for number of RAMs 
	 * @return number of RAMs
	 */
	public int getN_ram() {
		return n_ram;
	}

	/**	
	 * Setter for number of RAMs 
	 * @param n_ram number of RAMs
	 */
	public void setN_ram(int n_ram) {
		this.n_ram = n_ram;
	}
	/**	
	 * Getter for RAM array 
	 * @return array of RAMs
	 */
	public Ram[] getRams() {
		return rams;
	}
	/**	
	 * Getter for response (store) 
	 * @return response array of discriminator (double array of RAMs contents)
	 */
	public double[] getResponse() {
		return response;
	}
	/**
	 * Setter for mapping randomization seed
	 * @param seed seed for mapping randomization
	 */
	public void setSeed(long seed) {
		this.seed = seed;
	}

	/**
	 * Testing program
	 * @param args arguments
	 * @throws WisardException error on computation
	 */
	public static void main(String [] args) throws WisardException {
		try {
			Discriminator d = new Discriminator(2,8,"hello",mapType.RANDOM, -1);
			System.out.println(d.toString());
			long[] tuple = new long[4];
			tuple[0] = (long)0;
			tuple[1] = (long)1;
			tuple[2] = (long)1;
			tuple[3] = (long)0;
			d.train(tuple);
			System.out.println(d.toString());
			long[] tuple2 = new long[4];
			tuple2[0] = (long)1;
			tuple2[1] = (long)1;
			tuple2[2] = (long)1;
			tuple2[3] = (long)0;
			//d.train(tuple2);
			System.out.println(d.toString());
			System.out.println(String.format("Res: %f\n", d.classify(tuple2)));
			// Check histo
			double[] sample0 = {2,1,3};
			double[] sample1 = {2,2,3};
			double[] sample2 = {3,2,4};
			double[] sample3 = {4,0,4};
			double[] range = {4,4,4};
			double[] off = {0,0,0};
			int ntics = 3;
			int nfeatures = 3;
			Discriminator dd = new Discriminator(4,ntics * nfeatures,"hello",mapType.RANDOM, -1);
			dd.trainHisto(sample0,range,off,ntics,nfeatures);
			dd.trainHisto(sample1,range,off,ntics,nfeatures);
			dd.trainHisto(sample2,range,off,ntics,nfeatures);
			dd.trainHisto(sample3,range,off,ntics,nfeatures);
			System.out.println(dd.toString());
			double[] sampleT = {2,1,2};
			System.out.println(String.format("Res: %f\n", dd.classifyHisto(sampleT,range,off,ntics,nfeatures)));
			double maxmi = dd.updateMI();
			System.out.println(String.format("MI val = %f", maxmi));
			System.out.println(dd.MItoString());
		} catch (Exception exc) {
			System.out.println(exc);
			System.out.println("Problem in Wisard!");
		}
		
	}

}
