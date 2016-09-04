import functools
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
            'points': list(map(lambda point: (point['x'], point['y']), sample['points']))
        },
        plist_dict['samples']
    ))
    return cleaned_data


def add_neighboring_pairs(data):
    for sample in data:
        sample['pairs'] = []
        for current, nxt in zip(sample['points'][:], sample['points'][1:]):
            sample['pairs'].append((current, nxt))


def add_angles(data):
    for sample in data:
        sample['angles'] = []
        for pair in sample['pairs']:
            sample['angles'].append(
                math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0])))


def get_offset_angle_histogram(angles, bins, offset, norm=True):
    hist = numpy.histogram(list(map(lambda angle: loop_number(angle + offset, -180, 180), angles)),
                           bins=bins,
                           range=(-180, 180))[0]
    if norm:
        hist = hist / numpy.linalg.norm(hist)
    return hist


def get_point_area_histogram(coords, horizontal_divisions, vertical_divisions,
                             do_cells=False, do_rows=False, do_columns=False):
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


def write_bin_header(file, dataset_name, bins):
    file.write('@RELATION {}{}'.format(dataset_name, os.linesep))
    for bin in range(1, bins + 1):
        file.write('@ATTRIBUTE bin{} NUMERIC{}'.format(bin, os.linesep))

    file.write(
        '@ATTRIBUTE class {{{}}}{}@DATA{}'.format(','.join(map(chr, range(ord('A'), ord('Z') + 1))),
                                                  os.linesep,
                                                  os.linesep))


def main():
    data = transform_plist('samples.plist')
    add_neighboring_pairs(data)
    add_angles(data)
    output_directory = 'weka-files'

    if os.path.exists(output_directory):
        if os.path.isdir:
            shutil.rmtree(output_directory)
        else:
            raise Exception(output_directory + ' is not a directory')

    os.mkdir(output_directory)
    generate_and_write_vector_histogram(data, output_directory)


def loop_number(value, min, max):
    range = abs(min - max)
    while value > max:
        value -= range
    while value < min:
        value += range
    return value


def generate_and_write_vector_histogram(data, output_directory):
    for bins in [3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for offset in [(0, 'no'), (360 / bins / 2, 'half'), (360 / bins / 4, 'quarter')]:
            dataset_name = 'charReg-vectorHistogram-{:02d}bins-{}Offset'.format(bins, offset[1])
            file = open((os.path.join(output_directory, dataset_name) + '.arff'), 'w')
            write_bin_header(file, dataset_name, bins)
            for sample in data:
                histogram = get_offset_angle_histogram(sample['angles'], bins, offset[0])
                numpy_str_arr = numpy.char.mod('%f', histogram)
                file.write(','.join(numpy_str_arr))
                file.write(',' + sample['class'] + os.linesep)
            print('wrote ' + file.name)
            file.close()


if __name__ == '__main__':
    main()
    # plt.show()
