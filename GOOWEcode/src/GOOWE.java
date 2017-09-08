package GOOWE;
import MainPackage.run;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;

/**
 *
 * @author Hamed R. Bonab
 *  Date 17 March 2017
 */

public class GOOWE extends AbstractClassifier{
    
    // options 
    public ClassOption baseLearnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, 
			"trees.HoeffdingTree -e 2000000 -g 100 -c 0.01");
    public final boolean fuseOutput = true;
    
    public final int numOfHypo = run.globNumClassifiers;
    public final int numOfClasses = run.globNumClasses;
    
    final int fixedWindowPeriod = 500; //this specifies if no change in this period of time happens we should train new hypo and compare it with existings 
    
    public Classifier[] hypo; // array of classifiers in ensemble
    public double[] glob_weight;  // weights of each classifier in an ensemble
    
    InstanceList window;
    int num_proccessed_instance;
    int curNumOfHypo; //number of hypothesis till now     
    int candidateIndex;
    
    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        super.prepareForUseImpl(monitor, repository);
    }
    
    
    @Override
    public void resetLearningImpl() {
        window = new InstanceList(fixedWindowPeriod);
        this.num_proccessed_instance = 0;
        this.curNumOfHypo = 0;
        this.hypo = new Classifier[numOfHypo+1];
        glob_weight = new double[numOfHypo];
        for(int i=0; i< hypo.length ; i++){
            hypo[i] = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            hypo[i].resetLearning();
            hypo[i].prepareForUse();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
             
        double[][] votes = new double[curNumOfHypo][numOfClasses];
        for(int i=0; i<curNumOfHypo ; i++){
            double[] vote = normalizeVotes(hypo[i].getVotesForInstance(instnc));
            for(int j=0; j<vote.length ; j++){
                votes[i][j] = vote[j];
            }            
        }
        
        if(num_proccessed_instance<fixedWindowPeriod)
            votes = null;
        
        window.add(instnc, votes);
        this.num_proccessed_instance++;
                
        if(this.num_proccessed_instance % fixedWindowPeriod == 0){ //chunk is full
            processChunk();
        }
        
    }
    
    
    public double[] normalizeVotes(double[] votes){
        
        double[] newVotes = new double[numOfClasses];
        
        //check if all values are zero
        boolean allZero = true;
        for(int i=0; i<votes.length; i++){
            if(votes[i]>0)
                allZero=false;
        }
        
        
        if(allZero){ // all the votes are equal to zero
            
            double equalVote = 1.0/numOfClasses;
            for(int i=0; i<numOfClasses; i++){
                newVotes[i]=equalVote;
            }
            
        }else{ // votes are not equal to zero
            double sum=0;
            for(int i=0; i<votes.length; i++){
                sum+=votes[i];
            }
            for(int i=0; i<votes.length; i++){
                newVotes[i]=(votes[i]/sum);
            }
                    
        }
        
        return newVotes;                
        
    }
    
    //process a new given chunk of intances 
    private void processChunk() {
        //train new classifier on this new chunk
        for(int i=fixedWindowPeriod; i>0; i--){
            hypo[numOfHypo].trainOnInstance(window.getIns(i-1));
        }
        
        // weight and train new and rest classifiers 
        if(curNumOfHypo==0) { //there is no one
            candidateIndex = curNumOfHypo;
            hypo[candidateIndex] = (Classifier) hypo[numOfHypo].copy();
            glob_weight[candidateIndex] = 1.0;
            curNumOfHypo++;
            
        } else if (curNumOfHypo < numOfHypo) { //still has space 
            
            candidateIndex = curNumOfHypo;
            hypo[candidateIndex] = (Classifier) hypo[numOfHypo].copy();
            glob_weight[candidateIndex] = 1.0;
            double[] newWights = window.getWeight();
            for(int i=0; i<newWights.length; i++){                
                glob_weight[i] = newWights[i];                    
            }
            curNumOfHypo++;
            
        } else { // is full
            
            glob_weight = window.getWeight();
            //find minimum weight
            candidateIndex = 0;
            for(int i=1; i<glob_weight.length ; i++){
                if(glob_weight[i]<glob_weight[candidateIndex])
                    candidateIndex = i;
            }
            //substitutue
            hypo[candidateIndex] = (Classifier) hypo[numOfHypo].copy();
            glob_weight[candidateIndex] = 1.0;
        }

        //  train the rest of classifiers 
        for(int i=0;i<curNumOfHypo;i++){            
            //if(i==candidateIndex) // do not train candidate hypo again
            //    continue;            
            for(int j=0; j<fixedWindowPeriod; j++){
                hypo[i].trainOnInstance(window.getIns(j));
            }
        }
        
        hypo[numOfHypo].resetLearning();
    }
    
    
    @Override
    public boolean correctlyClassifies(Instance inst) {
        int expectedClass = Utils.maxIndex(getVotesForInstance(inst));
        int realClass = (int) inst.classValue();        
        
        return expectedClass == realClass;
    }
    
    
    @Override
    public double[] getVotesForInstance(Instance instnc) {
        DoubleVector combinedVote = new DoubleVector();        
        double[] hypo_weight = glob_weight; 
        
        for (int i = 0; i < curNumOfHypo; i++) {
            DoubleVector vote = new DoubleVector(hypo[i].getVotesForInstance(instnc));                        
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                vote.scaleValues(hypo_weight[i]);
                combinedVote.addValues(vote);
            }
        }        
        return combinedVote.getArrayRef();
    }
    
    
    public void printArray(double[] arr){
        for(int i=0 ; i<arr.length; i++){
            System.out.println(arr[i] + " "); 
        }
        System.out.println("");
    }
    
    @Override
    public boolean isRandomizable() {
        return true;
    }   
    
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   
    
}
