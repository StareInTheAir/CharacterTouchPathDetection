import functools
import multiprocessing
import os
import subprocess


def main():
    weka_jar = '/Applications/weka-3-8-0-oracle-jvm.app/Contents/Java/weka.jar'
    dataset_dir = 'weka-files'
    classifiers = ['weka.classifiers.bayes.NaiveBayes',
                   'weka.classifiers.functions.SMO',
                   'weka.classifiers.functions.MultilayerPerceptron',
                   # 'weka.classifiers.functions.Logistic',
                   'weka.classifiers.lazy.IBk',
                   'weka.classifiers.lazy.LWL',
                   'weka.classifiers.lazy.KStar',
                   'weka.classifiers.rules.DecisionTable',
                   'weka.classifiers.rules.JRip',
                   'weka.classifiers.rules.PART',
                   'weka.classifiers.trees.J48',
                   'weka.classifiers.trees.RandomForest',
                   'weka.classifiers.trees.RandomTree',
                   'weka.classifiers.trees.DecisionStump']

    classifiers = ['weka.classifiers.trees.RandomForest']

    output_file = open('weka-results.csv', 'w')
    csv_separator = ','

    csv_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=os.cpu_count())

    for classifier in classifiers:
        for dataset in os.listdir(dataset_dir):
            pool.apply_async(get_cv_accuracy,
                             (weka_jar, os.path.join(dataset_dir, dataset), classifier),
                             callback=functools.partial(sync_write_accuracy_to_csv, csv_lock,
                                                        output_file, classifier, dataset,
                                                        csv_separator))

    pool.close()
    pool.join()
    output_file.close()


def sync_write_accuracy_to_csv(lock, file, classifier, dataset, csv_separator, accuracy):
    print('{} on {} finished with {}%'.format(classifier, dataset, accuracy))
    lock.acquire()
    file.write(classifier + csv_separator + dataset + csv_separator + str(accuracy) + os.linesep)
    file.flush()
    lock.release()


def get_cv_accuracy(weka_jar, data, classifier):
    output = subprocess.check_output(
        'java -cp "{}" {} -t {} | '
        'grep -A 3 "Stratified" | '
        'grep "^Correctly" | '
        'awk \'{{print $5}}\''.format(
            weka_jar,
            classifier,
            data),
        shell=True)
    return float(output)


if __name__ == '__main__':
    main()
