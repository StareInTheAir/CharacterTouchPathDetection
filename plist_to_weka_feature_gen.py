import functools
import json
import math
import os
import plistlib
import shutil

import numpy

import geometry


def transform_plist(plist_path):
    plist_dict = plistlib.load(open(plist_path, 'rb'))
    cleaned_data = list(map(
        lambda sample:
        {
            'class': chr(ord('A') + sample['label']),
            'paths': [list(map(lambda point: (point['x'], point['y']), sample['points']))]
        },
        plist_dict['samples']
    ))
    return cleaned_data


def transform_json(json_path):
    json_dict = json.load(open(json_path))
    cleaned_data = list(map(
        lambda sample:
        {
            'class': sample['class'],
            'paths': list(
                map(lambda path: list(
                    map(lambda point: (point['x'], point['y']), path)
                ), sample['paths'])
            )
        },
        json_dict
    ))
    return cleaned_data


def add_neighboring_pairs(data):
    for sample in data:
        sample['pairs'] = []
        for path in sample['paths']:
            for current, nxt in zip(path[:], path[1:]):
            sample['pairs'].append((current, nxt))


def add_interpath_neighboring_pairs(data):
    for sample in data:
        paths = sample['paths']
        sample['interpath_pairs'] = []
        for path_index in range(len(paths)):
            # iterate through each path
            path = paths[path_index]
            for point_index in range(len(path)):
                # iterate through each point in path
                if point_index == len(path) - 1:
                    # path[point_index] is the last point in current path
                    if path_index != len(paths) - 1:
                        # path is not the last element in paths
                        current = path[point_index]
                        nxt = paths[path_index + 1][0]
                        sample['interpath_pairs'].append((current, nxt))
                else:
                    # path[point_index] isn't the last point in current path
                    current = path[point_index]
                    nxt = path[point_index + 1]
                    sample['interpath_pairs'].append((current, nxt))


def add_length(data):
    for sample in data:
        sample['lengths'] = []
        sample['interpath_lengths'] = []
        for vector in sample['pairs']:
            sample['lengths'].append(numpy.linalg.norm(vector))
        for vector in sample['interpath_pairs']:
            sample['interpath_lengths'].append(numpy.linalg.norm(vector))


def add_angles(data):
    for sample in data:
        sample['angles'] = []
        sample['interpath_angles'] = []
        for pair in sample['pairs']:
            sample['angles'].append(
                math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0])))
        for pair in sample['interpath_pairs']:
            sample['interpath_angles'].append(
                math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0])))


def get_offset_angle_histogram(angles, bins, offset, norm=True, lengths=None):
    hist = numpy.histogram(list(map(lambda angle: loop_number(angle + offset, -180, 180), angles)),
                           weights=lengths,
                           bins=bins,
                           range=(-180, 180))[0]
    if norm:
        hist = hist / numpy.linalg.norm(hist)
    return hist


def get_point_area_histogram(coords, horizontal_divisions, vertical_divisions,
                             do_cells=False, do_rows=False, do_columns=False,
                             norm=True):
    min_x = functools.reduce(lambda acc, cur: min(acc, cur[0]), coords, math.inf)
    max_x = functools.reduce(lambda acc, cur: max(acc, cur[0]), coords, -math.inf)
    min_y = functools.reduce(lambda acc, cur: min(acc, cur[1]), coords, math.inf)
    max_y = functools.reduce(lambda acc, cur: max(acc, cur[1]), coords, -math.inf)

    width = max_x - min_x
    height = max_y - min_y

    square_length = max(width, height)

    cell_width = square_length / horizontal_divisions
    cell_height = square_length / vertical_divisions

    cell_hist = []
    column_hist = []
    row_hist = []

    # fig = plt.figure()
    # ax = fig.add_subplot(111, aspect='equal')

    if do_cells:
        for horizontal_division in range(horizontal_divisions):
            for vertical_division in range(vertical_divisions):
                rect = get_rect(min_x + horizontal_division * cell_width,
                                min_y + vertical_division * cell_height,
                                cell_width,
                                cell_height)
                cell_hist.append(count_points_in_rect(coords, rect))

                # ax.add_patch(
                #     patches.Rectangle((min_x + horizontal_division * cell_width,
                #                        min_y + vertical_division * cell_height), cell_width,
                #                       cell_height, fill=False))

            if norm:
                cell_hist = norm_vector(cell_hist)

    if do_columns:
        for horizontal_division in range(horizontal_divisions):
            rect = get_rect(min_x + horizontal_division * cell_width,
                            min_y,
                            cell_width,
                            square_length)

            column_hist.append(count_points_in_rect(coords, rect))

            # ax.add_patch(
            #     patches.Rectangle((min_x + horizontal_division * cell_width, min_y),
            #                       cell_width, square_length, fill=False))

            if norm:
                column_hist = norm_vector(column_hist)

    if do_rows:
        for vertical_division in range(vertical_divisions):
            rect = get_rect(min_x,
                            min_y + vertical_division * cell_height,
                            square_length,
                            cell_height)
            row_hist.append(count_points_in_rect(coords, rect))

            # ax.add_patch(
            #     patches.Rectangle((min_x, min_y + vertical_division * cell_height),
            #                       square_length, cell_height, fill=False))

            if norm:
                row_hist = norm_vector(row_hist)

    # for coord in coords:
    #     ax.scatter(coord[0], coord[1])

    return cell_hist, column_hist, row_hist


def get_rect(x, y, width, height):
    return geometry.Rect(geometry.Point(x, y), geometry.Point(x + width, y + height))


def count_points_in_rect(coords, rect):
    bucket_value = 0
    for coord in coords:
        if rect.contains(coord):
            bucket_value += 1
    return bucket_value


def norm_vector(vector):
    norm = numpy.linalg.norm(vector).astype(float)
    return [x / norm for x in vector]


def arff_write_vector_angle_attributes(file, bins):
    for bin in range(1, bins + 1):
        arff_write_attribute(file, 'bin' + str(bin), 'NUMERIC')


def arff_write_point_area_attributes(file, divisions, histogram_combinations):
    if histogram_combinations[0]:
        for cell in range(1, divisions ** 2 + 1):
            arff_write_attribute(file, 'cell' + str(cell), 'NUMERIC')
    if histogram_combinations[1]:
        for row in range(1, divisions + 1):
            arff_write_attribute(file, 'row' + str(row), 'NUMERIC')
    if histogram_combinations[2]:
        for column in range(1, divisions + 1):
            arff_write_attribute(file, 'column' + str(column), 'NUMERIC')


def arff_write_relation(file, relation_name):
    file.write('@RELATION {}{}'.format(relation_name, os.linesep))


def arff_write_attribute(file, name, type):
    file.write('@ATTRIBUTE {} {}{}'.format(name, type, os.linesep))


def arff_write_data_start_marker(file):
    file.write('@DATA' + os.linesep)


def arff_write_alphabet_class_attribute(file):
    arff_write_attribute(file, 'class',
                         '{' + ','.join(map(chr, range(ord('A'), ord('Z') + 1))) + '}')


def loop_number(value, min, max):
    range = abs(min - max)
    while value > max:
        value -= range
    while value < min:
        value += range
    return value


def generate_and_write_vector_angle_histogram(data, output_directory):
    for bins in [3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for offset in [(0, 'no'), (360 / bins / 2, 'half'), (360 / bins / 4, 'quarter')]:
            dataset_name = 'charReg-vectorHistogram-{:02d}bins-{}Offset'.format(bins, offset[1])
            file = open((os.path.join(output_directory, dataset_name) + '.arff'), 'w')

            arff_write_relation(file, dataset_name)
            arff_write_vector_angle_attributes(file, bins)
            arff_write_alphabet_class_attribute(file)
            arff_write_data_start_marker(file)

            for sample in data:
                histogram = get_offset_angle_histogram(sample['angles'], bins, offset[0],
                                                       lengths=sample['lengths'])
                numpy_str_arr = numpy.char.mod('%f', histogram)
                file.write(','.join(numpy_str_arr))
                file.write(',' + sample['class'] + os.linesep)
            print('wrote ' + file.name)
            file.close()


def generate_and_write_point_area_histogram(data, output_directory):
    for divisions in [2, 3, 4, 5, 6, 8]:
        for histogram_combinations in [((True, False, False), 'cell'),
                                       ((False, True, True), 'rowColumn'),
                                       ((True, True, True), 'cellRowColumn')]:
            dataset_name = \
                'charReg-pointAreaHistogram-{}divisions-{}'.format(divisions,
                                                                   histogram_combinations[1])
            file = open((os.path.join(output_directory, dataset_name) + '.arff'), 'w')

            arff_write_relation(file, dataset_name)
            arff_write_point_area_attributes(file, divisions, histogram_combinations[0])
            arff_write_alphabet_class_attribute(file)
            arff_write_data_start_marker(file)

            for sample in data:
                (cell_hist, row_hist, column_hist) = \
                    get_point_area_histogram(sample['points'], divisions, divisions,
                                             do_cells=histogram_combinations[0][0],
                                             do_rows=histogram_combinations[0][1],
                                             do_columns=histogram_combinations[0][2])
                feature_sample = []

                if histogram_combinations[0][0]:
                    feature_sample += cell_hist
                if histogram_combinations[0][1]:
                    feature_sample += row_hist
                if histogram_combinations[0][2]:
                    feature_sample += column_hist

                file.write(','.join(map(str, feature_sample)))
                file.write(',' + sample['class'] + os.linesep)
            print('wrote ' + file.name)
            file.close()


def generate_and_write_best_combinations(data, output_directory):
    for bins in [8, 12]:
        for offset in [(360 / bins / 2, 'half'), (360 / bins / 4, 'quarter')]:
            for divisions in [4, 5, 6]:
                for histogram_combinations in [((True, False, False), 'cell'),
                                               ((True, True, True), 'cellRowColumn')]:
                    dataset_name = 'charReg-combinedHistogram-{}divisions-{}-{:02d}bins-{}Offset'. \
                        format(divisions, histogram_combinations[1], bins, offset[1])
                    file = open((os.path.join(output_directory, dataset_name) + '.arff'), 'w')

                    arff_write_relation(file, dataset_name)
                    arff_write_vector_angle_attributes(file, bins)
                    arff_write_point_area_attributes(file, divisions, histogram_combinations[0])
                    arff_write_alphabet_class_attribute(file)
                    arff_write_data_start_marker(file)

                    for sample in data:
                        vector_angle_hist = get_offset_angle_histogram(sample['angles'], bins,
                                                                       offset[0],
                                                                       lengths=sample['lengths'])
                        numpy_str_arr = numpy.char.mod('%f', vector_angle_hist)
                        file.write(','.join(numpy_str_arr))
                        file.write(',')

                        (cell_hist, row_hist, column_hist) = \
                            get_point_area_histogram(sample['points'], divisions, divisions,
                                                     do_cells=histogram_combinations[0][0],
                                                     do_rows=histogram_combinations[0][1],
                                                     do_columns=histogram_combinations[0][2])
                        point_area_histogram = []
                        if histogram_combinations[0][0]:
                            point_area_histogram += cell_hist
                        if histogram_combinations[0][1]:
                            point_area_histogram += row_hist
                        if histogram_combinations[0][2]:
                            point_area_histogram += column_hist
                        file.write(','.join(map(str, point_area_histogram)))

                        file.write(',' + sample['class'] + os.linesep)

                    print('wrote ' + file.name)
                    file.close()


def main():
    data = transform_plist('samples.plist')
    add_neighboring_pairs(data)
    add_length(data)
    add_angles(data)
    output_directory = 'weka-files'

    if os.path.exists(output_directory):
        if os.path.isdir:
            shutil.rmtree(output_directory)
        else:
            raise Exception(output_directory + ' is not a directory')

    os.mkdir(output_directory)
    # generate_and_write_vector_angle_histogram(data, output_directory)
    # generate_and_write_point_area_histogram(data, output_directory)
    generate_and_write_best_combinations(data, output_directory)


if __name__ == '__main__':
    main()
    # plt.show()
