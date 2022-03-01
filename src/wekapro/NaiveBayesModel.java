package wekapro;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesModel extends KnowledgeModel {
    NaiveBayes nbayes;

    public NaiveBayesModel() {
        super();
    }

    public NaiveBayesModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }

    public void buildNaiveBayes(String filename) throws Exception {
        // Doc train set vao bo nho
        setTrainset(filename);
        this.trainSet.setClassIndex(this.trainSet.numAttributes() -1);
        // huan luyen mo hinh Bayes
        this.nbayes = new NaiveBayes();
        // nbayes.setOptions(this.model_options);
        nbayes.buildClassifier(this.trainSet);
    }

    public void evaluatedNaiveBayes(String filename) throws Exception {
        // Doc file test vao bo nho
        setTestset(filename);
        this.testSet.setClassIndex(this.testSet.numAttributes() -1);
        // Danh gia mo hinh bang 10-fold cross-validation
        Random rand = new Debug.Random(1);
        int folds = 10;
        Evaluation eval = new Evaluation(this.trainSet);
        eval.crossValidateModel(nbayes, this.testSet, folds, rand);
        System.out.println(
                eval.toSummaryString("\nKet qua danh gia mo hinh 10-fold cross-validation\n-------------\n", false));
    }

    public void predictClassLabel(String fileIn, String fileOut) throws Exception {
        // Doc du lieu can du doan vao bo nho
        DataSource ds = new DataSource(fileIn);
        Instances unlabel = ds.getDataSet();
        unlabel.setClassIndex(unlabel.numAttributes() - 1);
        // Du doan classLable cho tung Instance
        for (int i = 0; i < unlabel.numInstances(); i++) {
            double prediet = nbayes.classifyInstance(unlabel.instance(i));
            unlabel.instance(i).setClassValue(prediet);
        }
        // Xuat ket qua ra fileout
        BufferedWriter outWriter = new BufferedWriter(new FileWriter(fileOut));
        outWriter.write(unlabel.toString());
        outWriter.newLine();
        outWriter.flush();
        outWriter.close();
    }

    @Override
    public String toString() {
        return this.nbayes.toString();
    }
}
