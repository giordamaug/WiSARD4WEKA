/*
 * **************************************************************************
 * Copyright 2018 Maurizio Giordano                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");          *
 * you may not use this file except in compliance with the License.         *
 * You may obtain a copy of the License at                                  *
 *                                                                          *
 * http://www.apache.org/licenses/LICENSE-2.0                               *
 *                                                                          *
 * Unless required by applicable law or agreed to in writing, software      *
 * distributed under the License is distributed on an "AS IS" BASIS,        *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 * See the License for the specific language governing permissions and      *
 * limitations under the License.                                           *
 ****************************************************************************/

package weka.classifiers.functions.wisard;

import java.io.Serializable;

import weka.core.RevisionHandler;

/**
 * <!-- globalinfo-start --> 
 * <b>Description:</b>
 * The class implementing the location (memory cell) of WiSARd RAMs.
 * A RAM location is a key-value pair: the key is the memory cell address, the value is its content.
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

public class Wentry implements Serializable, RevisionHandler {

    /** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;
	
	/** Cell key (address). */
	protected long key;                         
    /** Cell value (content). */
	protected double value;						

    /**
     * RAM's memory cell constructor
     * @param key key of cell
     * @param value conent of cell
     */
    public Wentry(long key, double value) {
        this.key = key;
        this.value = value;
    }
    
    
    /**
     * Getter of key of memory cell 
     * @return the cell key
     */
    public long getKey() {
        return key;
    }

    /**
     * Getter of content of memory cell 
     * @return the cell content
     */
    public double getValue() {
        return value;
    }

    /**
     * Setter of content of memory cell 
     * @param value the value to set in memory
     */
    public void setValue(double value) {
    	this.value = value;
    }

    /**
     * Updater of content of memory cell 
     * @param incr the value to add to the memory content
     */
    public void incrValue(double incr) {
    	this.value += incr;
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
