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
import java.util.Iterator;
import java.util.Vector;

import weka.core.RevisionHandler;

/**
 * <!-- globalinfo-start --> 
 * <b>Description:</b>
 * Implements the neuron/RAM object and the memory management system of WiSARD.
 * A RAM is a vector of key-value pairs: the key is the memory cell address, the value is its content.
 * key-value pairs are allocated (added to the vector) only when they content changes from zero to a nonzero value.
 * key-value pairs are deallocated (removed from the vector) when the content becomes zero.
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

public class Ram implements Serializable, RevisionHandler {
	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;
	/** Array of memory cells (key-value pairs). */
	protected Vector<Wentry> wentries = new Vector<Wentry>();     
	/** RAM index. */
	int index;                                                    

	/**
	 * Getter function of RAM key-value pair content given an input key
	 * @param key key to find the RAM key-value pair
	 * @return the key-value pair of RAM found by by key
	 */
	public Wentry getEntry (long key) { 
		Iterator<Wentry> it = (Iterator<Wentry>)this.wentries.iterator (); 
		while (it.hasNext ()) { 
			Wentry entry = (Wentry) it.next (); 
			if (key == entry.getKey()) { 
				return entry; 
			} 
		} 
		return (Wentry)null; 
	} 

	public void addKey(long key) {
		wentries.add(new Wentry(key,1.0));
	}

	/**
	 * Updater function of RAM content given an input key. The RAM content is incremented by 1 at each access.
	 * If the content was zero, the key-value pair is added to the RAM.
	 * @param key key to access the RAM location
	 * @return the content (after update) of RAM location accessed by key
	 */
	public double write(long key) {
		Wentry entry;
		if ((entry = (Wentry)getEntry((long)key)) == (Wentry)null) {
			this.addKey(key);
			return 1.0;
		} else {
			entry.incrValue(1.0);
			return entry.getValue();
		}	
	}

	/**
	 * Getter function of RAM location content given an input key
	 * @param key key to access the RAM location
	 * @return the content of RAM location accessed by key
	 */
	public double read(long key) {
		Wentry entry;
		if ((entry = (Wentry)getEntry(key)) == (Wentry)null) {
			return 0.0;
		} else {
			return wentries.get(index).value;
		}		
	}

	/**
	 * Print function of RAM contents
	 * @return the printout (string) of RAM contents
	 */
	public String toString() {
		String str = "{";

		Iterator<Wentry> it = this.wentries.iterator(); 
		while (it.hasNext()) {
			Wentry elem = it.next();
			str += String.format ( "%d:%f ", elem.key, elem.value);
		} 
		return str +  "}\n";
	}
	/**
	 * Getter of revision
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
}



