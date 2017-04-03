/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aiproject;
import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import static weka.core.Instances.test;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;
/**
 *
 * @author Duke
 */
public class AIPROJECT {
    static boolean DecisionTreeLoaded = false;
    static Instances trainingInstances;
    static J48 myJ48 = new J48();
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        
        System.out.println("Decision Tree Program: ");
        boolean QuitProgram = false;
        BufferedReader cnsl = new BufferedReader(new InputStreamReader(System.in));

        
        while(!QuitProgram)
        {
            System.out.printf("\nDecision Tree Loaded: %s\n\n1. Learn a decision tree or load an existing tree.\n2. Testing accuracy of the decision tree.\n3. Applying the decision tree to new cases.\n4. Quit\n\n>>", DecisionTreeLoaded);
            String consoleOutput = "";
            try {
                consoleOutput = cnsl.readLine();
            } catch (IOException ex) {
            }
            int decisionChoice = -1;
            try
            {
                decisionChoice = Integer.parseInt(consoleOutput);
            } 
            catch (Exception e)
            {
                System.out.print("Invalid Menu Option!");
                continue;
            }
            if(decisionChoice == 1)
            {
                learnLoadDecisionTree(cnsl);
            }
            else if(decisionChoice == 2)
            {
                if(DecisionTreeLoaded)
                {
                    testTreeAccuracy(cnsl);
                } else {
                    System.out.println("Please load a tree first before selecting this!");
                }
            }
            else if(decisionChoice == 3)
            {
                if(DecisionTreeLoaded)
                {
                    applyDecisionTreeToNewCases(cnsl);
                } else {
                    System.out.println("Please load a tree first before selecting this!");
                }
            }
            else if(decisionChoice == 4)
            {
                QuitProgram = true;
            }
            else {
                System.out.print("Invalid Menu Option!");
            }
        }
    }
    

    
    public static void learnLoadDecisionTree(BufferedReader myConsole) throws IOException
    {
        boolean breakloop = false;
        while(!breakloop)
        {
            System.out.printf("Please enter a file with attributes and training examples or a file with a stored tree:");
            String Output = myConsole.readLine();
            DataSource source;
            
            try
            {
                source = new DataSource(Output);
                trainingInstances = source.getDataSet();
        // Make the last attribute be the class
                if (trainingInstances.classIndex() == -1)
                {
                    trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
                }
                System.out.println("\nDataset:\n");
                System.out.println(trainingInstances);
                myJ48.buildClassifier(trainingInstances);
                DecisionTreeLoaded = true;
                
                breakloop = true;
            } 
            catch (Exception e)
            {
                
                System.out.printf("Invalid file: %s\n%s\n", Output, e.getMessage());
            }
        }
    }
    
    public static void testTreeAccuracy(BufferedReader myConsole) throws IOException, Exception
    {
        boolean breakloop = false;
        while(!breakloop)
        {
            System.out.printf("'q' - quit to mainmenu; Please enter a file with attributes and test examples:");
            String Output = myConsole.readLine();
            DataSource testDataSource;
            Instances testInstances;
            if(Output.length()== 1)
            {
                if(Output.toLowerCase().compareTo("q") == 0)
                {
                    return;
                } else {
                    System.out.println("Invalid input!");
                    continue;
                }
            }
            
            
            try
            {
                testDataSource = new DataSource(Output);
                testInstances = testDataSource.getDataSet();
                if (testInstances.classIndex() == -1)
                    testInstances.setClassIndex(testInstances.numAttributes() - 1);
                System.out.println(testInstances);
            } 
            catch (Exception e)
            {
                System.out.printf("Invalid file: %s\n%s", Output, e.getMessage());
                
                continue;
            }
                Evaluation eval = new Evaluation(testInstances);
                eval.evaluateModel(myJ48, testInstances);
                double[][] ConfusionMatrix = eval.confusionMatrix();
                
                
                
                for(int x = 0; x < ConfusionMatrix.length; x++)
                {
                    System.out.printf("\t%s",trainingInstances.classAttribute().value(x));
                }
                System.out.println();
                for(int x = 0; x < ConfusionMatrix.length; x++)
                {
                    for(int y = 0; y < ConfusionMatrix[0].length; y++)
                    {

                        System.out.printf("\t%.0f", ConfusionMatrix[x][y]);
                    }
                    System.out.println();
                }
                System.out.println();
        }
        
    }
    
    public static void applyDecisionTreeToNewCases(BufferedReader myConsole) throws IOException, Exception
    {
        boolean breakloop = false;
        while(!breakloop)
        {
            System.out.printf("\n1. Test a case interactively.\n2.Enter a new training case interactively. \n3.Quit\n\n>>");
            String Output = myConsole.readLine();
           
            int decisionChoice = -1;
            try
            {
                decisionChoice = Integer.parseInt(Output);
            } 
            catch (Exception e)
            {
                System.out.print("Invalid Menu Option!");
                continue;
            }
            if(decisionChoice == 2)
            {
                addAdditionalCase(myConsole);
                return;
            }
            else if(decisionChoice == 1)
            {
                testAnAdditionalCase(myConsole);
            }
            else if(decisionChoice == 3)
            {
                return;
            }
            else {
                System.out.print("Invalid Menu Option!");
            }
        }
    }
    
    /*1.  Learn a decision tree or load an existing tree.   
    2.  Testing accuracy of the decision tree. 
    3.  Applying the decision tree to new cases.
    The submenu of this item includes:    
        3.1  Enter a new case interactively.   
        3.2  Quit.  
    4. Save Decision Tree 
    4.
    Menu item 1:  prompt the user to enter file names of [attributes and training examples] or file names for stored trees. 
    Menu item 2:  prompt the user to enter testing data file name.  
    Output is the confusion matrix of applying a decision tree to the testing data set. 
    Menu item 3:  your program should guide the user to enter values of condition attributes interactively, 
    and allow the user to enter as many cases as desired. 
 */

  
    private static void addAdditionalCase(BufferedReader myReader) throws IOException, Exception  {
        boolean quitLoop = false;
        Instance originalInstanceDisplayed = trainingInstances.firstInstance();
        //trainingInstances.checkInstance(singleInstanceDisplayed)
        
        Instance singleInstanceDisplayed = (Instance)originalInstanceDisplayed.copy();
        while (!quitLoop) {
            System.out.printf("Currently Displayed String: %s\n", singleInstanceDisplayed.toString());
            System.out.printf("'q' to quit and save; 'c' to cancel; Please select value to edit from[1-%d]:\n", singleInstanceDisplayed.numAttributes());
            String Output = myReader.readLine();
            int decisionChoice = -1;
            try {
                decisionChoice = (Integer.parseInt(Output) - 1);
            } catch (Exception e) {
                if (Output.length() == 1) {
                    if(Output.toLowerCase().compareTo("q") == 0)
                    {
                        trainingInstances.add(singleInstanceDisplayed);
                        myJ48.buildClassifier(trainingInstances);
                        return;
                    } 
                    else if(Output.toLowerCase().compareTo("c") == 0)
                    {
                        return;
                    }
                }
                System.out.print("Invalid Option!");
                continue;
            }
            if(decisionChoice >= 0 && decisionChoice < singleInstanceDisplayed.numAttributes())
            {
                singleInstanceDisplayed = doSingleAttributeCaseEdit(singleInstanceDisplayed, decisionChoice, myReader); //if true then try to save single attribute edit
            } else {
                System.out.print("Invalid Option!");
            }
        }
    }

    private static Instance doSingleAttributeCaseEdit(Instance singleInstanceDisplayed, int decisionChoice, BufferedReader myReader) throws IOException, Exception{
        boolean quitLoop = false;
        System.out.flush();
        while (!quitLoop) {
            System.out.printf("Currently Displayed String: %s \n", singleInstanceDisplayed.toString());
            if(singleInstanceDisplayed.attribute(decisionChoice).isNumeric())
            {
                System.out.printf("\nq - quit and save changes; c - cancel changes. \nPlease enter a Numeric value for this: \n>>");
                String input = myReader.readLine();
                double selectedNumber = -1;
                try
                {
                    selectedNumber = Double.parseDouble(input);
                } 
                catch (Exception e)
                {
                    if(input.length() == 1)
                    {
                        if(input.toLowerCase().compareTo("q") == 0)
                        {
                            System.out.flush();                           
                            return singleInstanceDisplayed;
                        } 
                        else if(input.toLowerCase().compareTo("c") == 0)
                        {
                            System.out.flush();
                            return singleInstanceDisplayed;
                        }
                    }
                    System.out.println("Invalid Input!");
                    continue;
                }
                singleInstanceDisplayed.setValue(decisionChoice, selectedNumber);
                return singleInstanceDisplayed;
            } else if(singleInstanceDisplayed.attribute(decisionChoice).isNominal() || singleInstanceDisplayed.attribute(decisionChoice).isString()) {
                if(singleInstanceDisplayed.attribute(decisionChoice).isNominal())
                {
                    if(singleInstanceDisplayed.attribute(decisionChoice).numValues() < 10)
                    {
                        Enumeration<Object> myObject = singleInstanceDisplayed.attribute(decisionChoice).enumerateValues();
                        System.out.printf("Appropriate Values for this string:\n");
                        while(myObject.hasMoreElements())
                        {
                           System.out.printf(myObject.nextElement().toString() + " ");
                        }
                        System.out.println();
                    }
                }
                System.out.printf("q - quit and save changes; c - cancel changes. Please enter a String value for this: \n\n>>");
                String input = myReader.readLine();
                if(input.toLowerCase().compareTo("q") == 0)
                {
                    System.out.flush();                           
                    return singleInstanceDisplayed;
                } 
                else if(input.toLowerCase().compareTo("c") == 0)
                {
                    System.out.flush();
                    return singleInstanceDisplayed;
                }
                if(singleInstanceDisplayed.attribute(decisionChoice).isNominal())
                {
                    return checkAddIfNewNominal(singleInstanceDisplayed,singleInstanceDisplayed.attribute(decisionChoice), input, decisionChoice);
                }
                singleInstanceDisplayed.setValue(decisionChoice,input);
                return singleInstanceDisplayed;
            } else {
                System.out.printf("q - quit and save changes; c - cancel changes. Please enter a Date for this: \n\n>>");
                String input = myReader.readLine();
                if(input.toLowerCase().compareTo("q") == 0)
                {
                    System.out.flush();                           
                    return singleInstanceDisplayed;
                } 
                else if(input.toLowerCase().compareTo("c") == 0)
                {
                    System.out.flush();
                    return singleInstanceDisplayed;
                }
                try
                {
                    singleInstanceDisplayed.setValue(decisionChoice, singleInstanceDisplayed.attribute(decisionChoice).parseDate(input));
                    return singleInstanceDisplayed;
                } 
                catch (Exception e)
                {
                    continue;
                }
            }
        }
        return singleInstanceDisplayed;
    }

    private static Instance checkAddIfNewNominal(Instance singleInstanceDisplayed, Attribute aNominalAttribute, String attribute, int decisionChoice) throws Exception {
        //aNominalAttribute.
        //String[] Values = new String[aNominalAttribute.numValues() + 1];
        Instances[] myHack;
        StringBuilder Values = new StringBuilder();
        Enumeration<Object> aObject = aNominalAttribute.enumerateValues();
        int valCounter = 0;
        while(aObject.hasMoreElements())
        {
           String result = ((String)aObject.nextElement());
           if(Values.length() == 0)
           {
                Values.append(result);
           } else {
               Values.append("," + result);
           }
            if(result.toUpperCase().compareTo(attribute.toUpperCase()) == 0) //SAME ITEM! MUST CORRECT
            {
                singleInstanceDisplayed.setValue(decisionChoice, result);
                return singleInstanceDisplayed;
            }
        }
        //Unable to resolve Nominal
        //Adds original instance to data set
        trainingInstances.add(singleInstanceDisplayed);
        Values.append(",");
        Values.append(attribute);
        AddValues addValue = new AddValues();//adds attribute
        String Debug = Values.toString();
        addValue.setLabels(Debug);
        if(decisionChoice == 0)
        {
            addValue.setAttributeIndex("first");
        } 
        else
        {
            addValue.setAttributeIndex("" + (decisionChoice+1));
        }
        System.out.println(trainingInstances.numAttributes());
        //addValue.setAttributeIndex("" + (decisionChoice+1));
        addValue.setInputFormat(trainingInstances);
        System.out.println(trainingInstances.attribute(decisionChoice).toString().length());
        trainingInstances = Filter.useFilter(trainingInstances, addValue);
        System.out.println(trainingInstances.attribute(decisionChoice).toString().length());
        trainingInstances.lastInstance().setValue(decisionChoice, attribute);
        System.out.println(trainingInstances.lastInstance());
        singleInstanceDisplayed = trainingInstances.lastInstance();
        trainingInstances.remove(trainingInstances.numInstances()-1);
        return singleInstanceDisplayed;
    }

    private static void testAnAdditionalCase(BufferedReader myConsole) throws IOException {
        DenseInstance singleInstanceDisplayed = new DenseInstance(trainingInstances.numAttributes());
        singleInstanceDisplayed.setDataset(trainingInstances);
        boolean quitloop = false;
        while(!quitloop)
        {
        for(int i = 0; i < trainingInstances.numAttributes(); i++)
        {
            try {
                singleInstanceDisplayed = doSingleAttributeTestCaseEdit(singleInstanceDisplayed, i, myConsole); //if true then try to save single attribute edit
            } catch (Exception ex) {
                Logger.getLogger(AIPROJECT.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
            try {
                double Classification = myJ48.classifyInstance(singleInstanceDisplayed);
                Evaluation myEval = new Evaluation(singleInstanceDisplayed.dataset());
                System.out.print("Reported of class: " + trainingInstances.classAttribute().name() + " with value: " + trainingInstances.classAttribute().value((int)Classification));
             } catch (Exception ex) {
                 Logger.getLogger(AIPROJECT.class.getName()).log(Level.SEVERE, null, ex);
             }
            System.out.print("\n");
            boolean subLoop = false;
            while(!subLoop)
            {
                System.out.print("Would you like to enter another case? (Y/N) ");
                String input = myConsole.readLine();
                if(input.toUpperCase().compareTo("Y") == 0)
                {
                    break;
                }
                if(input.toUpperCase().compareTo("N") == 0)
                {
                    return;
                }
            }
        }
    }

   public static void testCaseOutputLine(String AttributeName, String AttributeType)
   {
       System.out.printf("Please enter the value for %s which is type %s", AttributeName, AttributeType);
   }

   
 private static DenseInstance doSingleAttributeTestCaseEdit(DenseInstance singleInstanceDisplayed, int decisionChoice, BufferedReader myReader) throws IOException, Exception{
        boolean quitLoop = false;
        System.out.flush();
        while (!quitLoop) {
            System.out.printf("Currently Displayed String: %s \n", singleInstanceDisplayed.toString());
            if(singleInstanceDisplayed.attribute(decisionChoice).isNumeric())
            {
                System.out.printf("\n\nPlease enter a Numeric value for this: \n>>");
                String input = myReader.readLine();
                double selectedNumber = -1;
                try
                {
                    selectedNumber = Double.parseDouble(input);
                } 
                catch (Exception e)
                {
                    System.out.println("Invalid Input!");
                    continue;
                }
                singleInstanceDisplayed.setValue(decisionChoice, selectedNumber);
                return singleInstanceDisplayed;
            } else if(singleInstanceDisplayed.attribute(decisionChoice).isNominal() || singleInstanceDisplayed.attribute(decisionChoice).isString()) {
                if(singleInstanceDisplayed.attribute(decisionChoice).isNominal())
                {
                    if(singleInstanceDisplayed.attribute(decisionChoice).numValues() < 10)
                    {
                        Enumeration<Object> myObject = singleInstanceDisplayed.attribute(decisionChoice).enumerateValues();
                        System.out.printf("Appropriate Values for this string:\n");
                        while(myObject.hasMoreElements())
                        {
                           System.out.printf(myObject.nextElement().toString() + " ");
                        }
                        System.out.println();
                    }
                }
                System.out.printf("Please enter a String value for this: \n\n>>");
                String input = myReader.readLine();
                
                if(singleInstanceDisplayed.attribute(decisionChoice).isNominal())
                {
                    return checkAddIfNewNominal(singleInstanceDisplayed,singleInstanceDisplayed.attribute(decisionChoice), input, decisionChoice);
                }
                singleInstanceDisplayed.setValue(decisionChoice,input);
                return singleInstanceDisplayed;
            } else {
                System.out.printf("Please enter a Date for this: \n\n>>");
                String input = myReader.readLine();
                if(input.toLowerCase().compareTo("q") == 0)
                {
                    System.out.flush();                           
                    return singleInstanceDisplayed;
                } 
                else if(input.toLowerCase().compareTo("c") == 0)
                {
                    System.out.flush();
                    return singleInstanceDisplayed;
                }
                try
                {
                    singleInstanceDisplayed.setValue(decisionChoice, singleInstanceDisplayed.attribute(decisionChoice).parseDate(input));
                    return singleInstanceDisplayed;
                } 
                catch (Exception e)
                {
                    continue;
                }
            }
        }
        return singleInstanceDisplayed;
    }
   
    private static DenseInstance checkAddIfNewNominal(DenseInstance singleInstanceDisplayed, Attribute aNominalAttribute, String attribute, int decisionChoice) throws Exception {
        //aNominalAttribute.
        //String[] Values = new String[aNominalAttribute.numValues() + 1];
        Instances[] myHack;
        StringBuilder Values = new StringBuilder();
        Enumeration<Object> aObject = aNominalAttribute.enumerateValues();
        int valCounter = 0;
        while(aObject.hasMoreElements())
        {
           String result = ((String)aObject.nextElement());
           if(Values.length() == 0)
           {
                Values.append(result);
           } else {
               Values.append("," + result);
           }
            if(result.toUpperCase().compareTo(attribute.toUpperCase()) == 0) //SAME ITEM! MUST CORRECT
            {
                singleInstanceDisplayed.setValue(decisionChoice, result);
                return singleInstanceDisplayed;
            }
        }
        //Unable to resolve Nominal
        //Adds original instance to data set
        trainingInstances.add(singleInstanceDisplayed);
        Values.append(",");
        Values.append(attribute);
        AddValues addValue = new AddValues();//adds attribute
        String Debug = Values.toString();
        addValue.setLabels(Debug);
        if(decisionChoice == 0)
        {
            addValue.setAttributeIndex("first");
        } 
        else
        {
            addValue.setAttributeIndex("" + (decisionChoice+1));
        }
        System.out.println(trainingInstances.numAttributes());
        //addValue.setAttributeIndex("" + (decisionChoice+1));
        addValue.setInputFormat(trainingInstances);
        System.out.println(trainingInstances.attribute(decisionChoice).toString().length());
        trainingInstances = Filter.useFilter(trainingInstances, addValue);
        System.out.println(trainingInstances.attribute(decisionChoice).toString().length());
        trainingInstances.lastInstance().setValue(decisionChoice, attribute);
        System.out.println(trainingInstances.lastInstance());
        singleInstanceDisplayed = (DenseInstance)trainingInstances.lastInstance();
        trainingInstances.remove(trainingInstances.numInstances()-1);
        return singleInstanceDisplayed;
    }         
    
    
    
    
    
    
    
}
