/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package wekapro;
//import weka.classifiers.trees.J48;
/**
 *
 * @author KameDev
 */
public class WekaPro {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // AprioriModel model = new AprioriModel("/Users/kamedev/Documents/Code/Java/data/supermarket.arff", "-N 50 -T 0 -C 0.8 -D 0.05 -U 1.0 -M 0.3 -S -1.0 -c -1", "-R 1-9,11,57,70,79-81,88-89,98,100-102,107-114,116-120,122-130,137-179,189,192-199,201-216");

        // FPGrowthModel model = new FPGrowthModel("/Users/kamedev/Documents/Code/Java/data/supermarket.arff", "-P 2 -I -1 -N 50 -T 0 -C 0.8 -D 0.05 -U 1.0 -M 0.3","-R 1-9,11,57,70,79-81,88-89,98,100-102,107-114,116-120,122-130,137-179,189,192-199,201-216");
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

        DecisionTreeModel model = new DecisionTreeModel("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris.arff", "-C 0.25 -M 2", null);
        model.buildDecisionTree();
        model.evaluatedDecisionTree();
        System.out.println(model);
        // model.saveModel("/Users/kamedev/Documents/Code/Java/decisionModel", model.tree);
        // model.tree = (J48)model.loadModel("/Users/kamedev/Documents/Code/Java/decisionModel");
        // model.predictClassLabel(model.testSet);

        // NaiveBayesModel model = new NaiveBayesModel();
        // model.buildNaiveBayes("/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_train.arff");
        // model.evaluatedNaiveBayes("/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_test.arff");
        // model.predictClassLabel("/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_unlabel.arff", "/Users/kamedev/Documents/Code/Java/WekaPro/data/labor_predict_nb.arff");
        // System.out.println(model);

        // DecisionTreeModel model = new DecisionTreeModel();
        // model.buildRandomForest("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_train.arff");
        // model.evaluatedRandomForest("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_test.arff");
        // model.predictClassLabel("/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_unlabel.arff", "/Users/kamedev/Documents/Code/Java/WekaPro/data/iris_predict_nb.arff");
        // System.out.println(model);
    }
}
