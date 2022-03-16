/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package wekapro;
import weka.classifiers.trees.J48;
/**
 *
 * @author KameDev
 */
public class WekaPro {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // AprioriModel model = new AprioriModel("/Users/kamedev/Downloads/DataMining/baitap2va3nhom1.arff", "-N 20 -T 0 -C 1.0 -D 0.05 -U 1.0 -M 0.5 -S -1.0 -c -1", null);

        // FPGrowthModel model = new FPGrowthModel("/Users/kamedev/Downloads/DataMining/baitap2va3nhom1.arff", "-P 2 -I -1 -N 20 -T 0 -C 1.0 -D 0.05 -U 1.0 -M 0.5", null);
        // model.mineAssociationRules();
        // System.out.println(model);

        // KnowledgeModel model = new
        // KnowledgeModel("/Users/kamedev/Documents/Code/Java/data/iris.arff", null,
        // null);
        // model.trainSet = model.divideTrainTestR(model.wekaDataset, 80, false);
        // model.testSet = model.divideTrainTestR(model.wekaDataset, 20, true);
        // System.out.println(model);
        // System.out.println(model.trainSet.toSummaryString());
        // System.out.println(model.testSet.toSummaryString());
        // model.saveDataToCSV("C:\\Users\\KameDev\\Documents\\ARFF\\vidux.csv");

        DecisionTreeModel model = new DecisionTreeModel("/Users/kamedev/Downloads/DataMining/baitap2cayquyetdinh.arff", "-C 0.25 -M 2", null);
        model.buildDecisionTree();
        model.evaluatedDecisionTree();
        System.out.println(model);
        // model.saveModel("/Users/kamedev/Documents/Code/Java/decisionModel", model.tree);
        // model.tree = (J48)model.loadModel("/Users/kamedev/Documents/Code/Java/decisionModel");
        // model.predictClassLabel(model.testSet);

        // NaiveBayesModel model = new NaiveBayesModel();
        // model.buildNaiveBayes("/Users/kamedev/Documents/Code/Java/WekaPro/data/trainset_70.arff");
        // model.evaluatedNaiveBayes("/Users/kamedev/Documents/Code/Java/WekaPro/data/testset_30.arff");
        // model.predictClassLabel("/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_unlabel.arff", "/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_predict_nb.arff");
        // System.out.println(model);

        // DecisionTreeModel model = new DecisionTreeModel();
        // model.buildRandomForest("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_train.arff");
        // model.evaluatedRandomForest("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_test.arff");
        // model.predictClassLabel("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_unlabel.arff", "/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_predict_nb.arff");
        // System.out.println(model);
    }
}
