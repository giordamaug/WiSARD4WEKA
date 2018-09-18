/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Copyright 2002 University of Waikato
 */

package weka.classifiers.functions;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;

// call as:
// > java -cp /Users/maurizio/Work/WEKA-3-8-3/weka-3-8-3/weka.jar:dist/WiSARD-tests.jar:lib/weka-stable-3.8.3-tests.jar:lib/junit.jar:dist/WiSARD.jar weka.classifiers.functions.WiSARDTest
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
