/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package wekapro;

import weka.classifiers.evaluation.*;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Debug.Random;

public class DecisionTreeModel extends KnowledgeModel{
    J48 tree;

    public DecisionTreeModel(){

    }
    
    public DecisionTreeModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    public void buildDecisionTree() throws Exception{
        //Tao tap du lieu train test
        this.trainSet = divideTrainTestR(this.wekaDataset, 80, false);
        this.testSet = divideTrainTestR(this.wekaDataset, 80, true);

        this.trainSet.setClassIndex(this.trainSet.numAttributes()-1);
        this.testSet.setClassIndex(this.testSet.numAttributes()-1);
        //Thiet lap thong so cho mo hinh cay quyet dinh
        tree = new J48();
        tree.setOptions(this.model_options);
        //Huan luyen mo hinh voi tap du lieu train
        tree.buildClassifier(this.trainSet);
    }

    public void evaluatedDecisionTree() throws Exception{
        Random rand = new Debug.Random(1);
        int folds = 10;
        Evaluation eval = new Evaluation(this.trainSet);
        eval.crossValidateModel(tree, this.testSet, folds, rand);
        //eval.evaluateModel(tree, this.testSet);
        System.out.println(eval.toSummaryString("\nKet qua danh gia mo hinh 10-folds cross-validation...\n----------\n", false));
    }
    
    @Override
    public String toString() {
        return tree.toSummaryString();
    }

    public void predictClassLabel(Instances input) throws Exception{
        for(int i = 0; i < input.numInstances(); i++){
            double predict = tree.classifyInstance(input.instance(i));
            double actual = input.instance(i).classValue();

            System.out.println("Instance " + i + ": predict = " + input.classAttribute().value((int)predict) + ";\t actual = " + input.classAttribute().value((int)actual));

            // input.instance(i).setClassValue(predict);
        }
    }
}