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


/*
 * Copyright 2002 University of Waikato
 */

package weka.classifiers.functions;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests WiSARD. Run from the command line with:<p>
 * java weka.classifiers.functions.WiSARDTest
 *
 * @author <a href="mailto:eibe@cs.waikato.ac.nz">Eibe Frank</a>
 * @version $Revision: 1.2 $
 */
public class WiSARDTest extends AbstractClassifierTest {

  public WiSARDTest(String name) { 
	  super(name); 
	  // DEBUG = true; 
  }

  /** Creates a default Winnow */
  @Override
  public Classifier getClassifier() {
    return new WiSARD();
  }
  /**
   * Skip regression tests. Only classification with nominal classes is allowed
   */
  @Override
  public void testRegression() throws Exception {
	  return;
  }
  public static Test suite() {
    return new TestSuite(WiSARDTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }

}
