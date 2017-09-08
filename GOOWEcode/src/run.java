package MainPackage;

import GOOWE.*;
import Baselines.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Scanner;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.streams.ArffFileStream;
import sizeof.agent.SizeOfAgent;

/**
 *
 * @author Hamed R. Bonab
 * Date 17 March 2017
 */

public class run {
    public static int globNumClassifiers = 2;
    public static int globNumClasses;
    
    public static String globNameOfDS;
    public static int globNumFeatures;
    public static int fixedWindowPeriod = 500;
    
    public static int index =0;
    
    public static void main(String[] args) throws FileNotFoundException, IOException  {
        // read the dataset folder and obtain each file 
        String dir = "dataset";
        String outdir = "output//test-";
        File directory = new File(dir);
        
        int[] ncOptions = { 2, 4 }; //number of classifiers of ensemble options 
        
        String outFileName = outdir + "acc--output" + ".txt";
        BufferedWriter writer =  new BufferedWriter(new FileWriter(new File(outFileName)));
            
        for (File file : directory.listFiles()) 
        {
            // write the result into a file accuracy + time + memory  
            String outCSV = "";
            
            for(int i=0; i<ncOptions.length; i++)
            {
                globNumClassifiers = ncOptions[i];
                int max_limit = 10000;  // maximum number of instances to be considered
                
                //preparing dataset
                index = 0;
                // Data Stream perparation
                ArffFileStream stream = new ArffFileStream(file.getAbsolutePath(), -1);
                // if the class label is not the latest feature we do specify it here 
                if(file.getName().equalsIgnoreCase("real-Click_prediction_small.arff") || 
                        file.getName().equalsIgnoreCase("real-covtype-binary.arff") ){
                    stream.classIndexOption.setValue(1);
                }
                stream.prepareForUse();
                
                globNameOfDS = stream.getHeader().getRelationName();
                globNumFeatures = stream.getHeader().numAttributes(); 
                globNumClasses = stream.getHeader().numClasses();
                
                System.out.println("Name of DS " + globNameOfDS);
                
                //preparing learner 
                //GOOWE learner = new GOOWE();
                myAUE2 learner = new myAUE2();
                //learner.ensembleSizeOption.setValue(globNumClassifiers);
                learner.setModelContext(stream.getHeader());
                learner.prepareForUse();

                long starttime, endtime;
                starttime = System.currentTimeMillis();
                int numOfCorrect = 0;
                long memMax = 0;
                
                while(stream.hasMoreInstances() && index<=max_limit){
                    try{
                        //System.out.println("index " + index);
                        InstanceExample inst = stream.nextInstance();

                        if(index>((globNumClasses+1)*(fixedWindowPeriod))){
                            //System.out.println("index " + index);
                            if(learner.correctlyClassifies(inst.instance)){
                                numOfCorrect++;
                            }
                            
                        } 
                        
                        index++;
                        learner.trainOnInstance(inst.instance);
                        
                        //long mem = SizeOfAgent.fullSizeOf(learner);
                        //if(mem > memMax )
                        //    memMax = mem;
                       
                    }catch(Exception e){
                       writer.write(e.getMessage());
                       System.out.println(e.getMessage());
                       break;
                    }

                }
              
                endtime = System.currentTimeMillis();
                
                double accuracy = (numOfCorrect/(double)index)*100;
                accuracy = round(accuracy, 3);

                //writer.write("dataset : " + file.getName()+ ";  nc= " + globNumClassifiers + ";\n");
                //writer.write("Acc : " + accuracy + "\n ####################\n\n");

                System.out.println("dataset : " + file.getName() + ";  nc= " + globNumClassifiers + ";\n");
                System.out.println("Acc AUE: " + accuracy + " | Time: "+(endtime - starttime) + " ms  \n");
                
                
            }// end of ncoptions for
            //writer.write(outCSV);
            
        }// end of list of file for 
        
            writer.close();
    }
    
    public static double round(double value, int places) {
        if (places < 0) 
            throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
