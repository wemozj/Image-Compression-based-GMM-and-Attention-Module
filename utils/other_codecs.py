"""
implement other codecs(modes):
* JPEG(jp)
* JPEG2k(jp2k)
* Webp(webp)
* TODO: PNG to plan(png)

flow:


"""

import argparse
import csv
import numpy as np
import glob
import time
import scipy.ndimage
import shutil
import os
import sys
import subprocess
from contextlib import contextmanager
import operator
import re
from PIL import Image
import itertools
import functools
import matplotlib.pyplot as plt

from utils import compare_imgs

# call environ variable
# if KDU_COMPRESS in system env variable, return value;else, return kud_compress
KDU_COMPRESS = os.environ.get('KDU_COMPRESS', 'kdu_compress')

CWEBP = os.environ.get('CWEBP', 'cwebp')
DWEBP = os.environ.get('DWEBP', 'dwebp')

BPGENC = os.environ.get('BPGENC', 'bpgenc')

_BPG_QUANTIZATION_PARAMETER_RANGE = (1, 51)  # smaller means better
_KDU_RE_PAT = r'Compressed bytes \(excludes codestream headers\) = .*=\s(.*)\sbpp'

SUPPORTED_METRICS = ('psnr', 'ssim', 'ms-ssim')
# other ----------------------------------------------
def read_measures(bpg_image_csv, metric):
    assert metric in SUPPORTED_METRICS
    with open(bpg_image_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # first field is q for BPG and JP, target_bpp for JP2K
        for _, bpp, ssim, msssim, psnr in reader:
            yield float(bpp), float({'ssim': ssim, 'ms-ssim': msssim, 'psnr': psnr}[metric])


def all_measures_file_ps(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, '*_out.csv')))


def _get_image_paths(root_dir):
	if '*' in root_dir:  # assume glob
		print('Assuming glob {}'.format(root_dir))
		return sorted(glob.glob(root_dir))

	_, ext = os.path.splitext(root_dir)
	if ext != '':  # assume single image
		print('Assuming single image with extenstion {}'.format(ext))
		return [root_dir]

	print('Assuming folder of PNGs: {}'.format(root_dir))
	return sorted(glob.glob(os.path.join(root_dir, '*.png')))


class BinarySearchFailedException(Exception):
	def __init__(self, discovered_values):
		self.discovered_values = discovered_values

	def first_x_yielding_y_greater_than(self, y_target):
		for x, y in sorted(self.discovered_values, key=operator.itemgetter(1)):
			if y > y_target:
				return x
		raise ValueError('No x found with y > {} in {}.'.format(y_target, self.discovered_values))


def binary_search(f, g, f_type, y_target, y_target_eps, x_min, x_max, x_eps, max_num_iter=1000, log=True):
	""" does binary search on f :: X -> Z by calculating z = f(x) and using g :: Z -> Y to get y = g(z) = g(f(x)).
	(g . f) is assumed to be monotonically increasing iff f_tpye == 'increasing' and monotonically decreasing iff
	f_type == 'decreasing'.
	Returns first (x, z) for which |y_target - g(f(x))| < y_target_eps. x_min, x_max specifiy initial search interval for x.
	Stops if x_max - x_min < x_eps. Raises BinarySearchFailedException when x interval too small or if search takes
	more than max_num_iter iterations. The expection has a field `discovered_values` which is a list of checked
	(x, y) coordinates. """

	def _print(s):
		if log:
			print(s)

	assert f_type in ('increasing', 'decreasing')
	cmp_op = operator.gt if f_type == 'increasing' else operator.lt
	discovered_values = []
	print_col_width = len(str(x_max)) + 3
	for _ in range(max_num_iter):
		x = x_min + (x_max - x_min) / 2
		z = f(x)
		y = g(z)
		discovered_values.append((x, y))
		_print('[{:{width}.2f}, {:{width}.2f}] -- g(f({:{width}.2f})) = {:.2f}'.format(
			x_min, x_max, x, y, width=print_col_width))
		if abs(y_target - y) < y_target_eps:
			return z, x
		if cmp_op(y, y_target):
			x_max = x
		else:
			x_min = x
		if x_max - x_min < x_eps:
			_print('Stopping, interval too close!')
			break
	sorted_discovered_values = sorted(discovered_values)
	first_y, last_y = sorted_discovered_values[0][1], sorted_discovered_values[-1][1]
	if (f_type == 'increasing' and first_y > last_y) or (f_type == 'decreasing' and first_y < last_y):
		raise ValueError('Got f_type == {}, but first_y, last_y = {}, {}'.format(
			f_type, first_y, last_y))
	raise BinarySearchFailedException(discovered_values)


def _append_to_measures_f(f, q, bpp, ssim, msssim, psnr):
	fout_str = ','.join(map('{:.3f}'.format, [q, bpp, ssim, msssim, psnr]))
	f.write(fout_str + '\n')
	return fout_str


def check_if_program_is_available(prg, name, env_name):
	try:
		subprocess.call([prg, '-v'],
						stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except FileNotFoundError:
		print("*** Invalid path to {}: {}".format(name, prg))
		print('Make sure {} is available in $PATH or at ${}.'.format(name, env_name))
		sys.exit(1)


@contextmanager
def remove_file_after(p):
	yield p
	os.remove(p)


def convert_im_to(ext, input_image_p):
	input_image_root_p, _ = os.path.splitext(input_image_p)
	im = Image.open(input_image_p)
	input_image_ext_p = input_image_root_p + '__tmp.{}'.format(ext)
	im.save(input_image_ext_p)
	return input_image_ext_p


def measures_file_p(bpg_dir, img_name):
	return os.path.join(bpg_dir, '{}_out.csv'.format(img_name))


# ------------------ABOUT BPG------------------------------------------
# BPG compress
def bpg_compress(input_image_p, q, tmp_dir=None, chroma_fmt='444'):
	""" Int -> image_out_path :: str """
	assert 'png' in input_image_p
	if tmp_dir:
		input_image_name = os.path.basename(input_image_p)
		output_image_bpg_p = os.path.join(tmp_dir, input_image_name).replace('.png', '_tmp_bpg.bpg')
	else:
		output_image_bpg_p = input_image_p.replace('.png', '_tmp_bpg.bpg')
	subprocess.call([BPGENC, '-q', str(q), input_image_p, '-o', output_image_bpg_p, '-f', chroma_fmt])
	return output_image_bpg_p


# bpg precise Bitrates
def bpg_measure(input_image_p, bpp, precise=False, save_output_as_png=None, tmp_dir=None):
	"""
	:return (PSNR, SSIM, MS-SSIM, actual_bpp)
	"""
	input_image_root_p, ext = os.path.splitext(input_image_p)
	assert ext == '.png', 'Expected PNG to convert to BMP, got {}'.format(input_image_p)
	output_image_bpg_p, actual_bpp = _bpg_compress_to_bpp(
		input_image_p, target_bpp=bpp, precise=precise)
	output_image_bpg_png_p = decode_bpg_to_png(output_image_bpg_p)
	os.remove(output_image_bpg_p)  # don't need that anymore

	_, msssim, _ = compare_imgs.compare(input_image_p, output_image_bpg_png_p,
										calc_ssim=False, calc_msssim=True, calc_psnr=False)
	if save_output_as_png:
		os.rename(output_image_bpg_png_p, save_output_as_png)
	else:
		os.remove(output_image_bpg_png_p)
	return msssim, actual_bpp


def _bpg_compress_to_bpp(input_image_p, target_bpp, precise=False, tmp_dir=None):
	def compress_input_image_with_quality(q):
		return bpg_compress(input_image_p, q, tmp_dir)

	bpp_eps = 0.01 if precise else 0.05
	try:
		q_min, q_max = _BPG_QUANTIZATION_PARAMETER_RANGE
		output_image_bpg_p, q = binary_search(compress_input_image_with_quality, bpp_of_bpg_image, 'decreasing',
											  y_target=target_bpp, y_target_eps=bpp_eps,
											  x_min=q_min, x_max=q_max, x_eps=0.1, log=False)
	except BinarySearchFailedException as e:
		q = e.first_x_yielding_y_greater_than(target_bpp)
		output_image_bpg_p = compress_input_image_with_quality(q)

	print('q = {}'.format(q))
	actual_bpp = bpp_of_bpg_image(output_image_bpg_p)
	return output_image_bpg_p, actual_bpp


def decode_bpg_to_png(bpg_p):  # really fast
	png_p = bpg_p.replace('.bpg', '_as_png.png')
	subprocess.call(['bpgdec', '-o', png_p, bpg_p])
	return png_p


def bpp_of_bpg_image(bpg_p):
	return bpg_image_info(bpg_p).bpp


class BPGImageInfo(object):
	def __init__(self, width, height, num_bytes_for_picture):
		self.width = width
		self.height = height
		self.num_bytes_for_picture = num_bytes_for_picture
		self.bpp = num_bytes_for_picture * 8 / float(width * height)


def bpg_image_info(p):
	"""
	Relevant format spec:
	magic number          4 bytes
	header stuff          2 bytes
	width                 variable, ue7
	height                variable, ue7
	picture_data_length   variable, ue7. If zero: remaining data is image
	"""
	with open(p, 'rb') as f:
		magic = f.read(4)
		expected_magic = bytearray.fromhex('425047fb')
		assert magic == expected_magic, 'Not a BPG file it seems: {}'.format(p)
		header_info = f.read(2)
		width = _read_ue7(f)
		height = _read_ue7(f)
		picture_data_length = _read_ue7(f)
		num_bytes_for_picture = _number_of_bytes_until_eof(f) if picture_data_length == 0 else picture_data_length
		return BPGImageInfo(width, height, num_bytes_for_picture)


def _read_ue7(f):
	"""
	ue7 means it's a bunch of bytes all starting with a 1 until one byte starts
	with 0. from all those bytes you take all bits except the first one and
	merge them. E.G.

	some ue7-encoded number:      10001001 01000010
	take all bits except first ->  0001001  1000010
	merge ->                            10011000010 = 1218
	"""
	bits = 0
	first_bit_mask = 1 << 7
	value_holding_bits_mask = int(7 * '1', 2)
	for byte in _byte_generator(f):
		byte_as_int = byte[0]
		more_bits_are_coming = byte_as_int & first_bit_mask
		bits_from_this_byte = byte_as_int & value_holding_bits_mask
		bits = (bits << 7) | bits_from_this_byte
		if not more_bits_are_coming:
			return bits


def _number_of_bytes_until_eof(f):
	return sum(1 for _ in _byte_generator(f))


def _byte_generator(f):
	while True:
		byte = f.read(1)
		if byte == b"":
			break
		yield byte


def bpg_measure_over_interval(input_image_p, fout, grid):
	input_image_root_p, ext = os.path.splitext(input_image_p)
	assert ext == '.png', 'Expected PNG to convert to BMP, got {}'.format(input_image_p)

	for q in map(int, grid):
		with remove_file_after(bpg_compress(input_image_p, q)) as out_p:
			bpp = bpp_of_bpg_image(out_p)
			with remove_file_after(decode_bpg_to_png(out_p)) as out_png_p:
				im_in = scipy.ndimage.imread(input_image_p)
				im_out = scipy.ndimage.imread(out_png_p)
				ssim, msssim, psnr = compare_imgs.compare(
					im_in, im_out, calc_ssim=True, calc_msssim=True, calc_psnr=True)
				fout_str = _append_to_measures_f(fout, q, bpp, ssim, msssim, psnr)
				print(fout_str, end='\r')
	print()


# ----------------------ABOUT JP---------------------------------
def jp_compress(input_image_p, q):
	output_image_jp_p = os.path.splitext(input_image_p)[0] + '_out_jp_{}.jpeg'.format(q)
	img = Image.open(input_image_p)
	img.save(output_image_jp_p, quality=q, subsampling=0)
	dim = float(np.prod(img.size))
	bpp = (8 * _jpeg_content_length(output_image_jp_p)) / dim
	return bpp, output_image_jp_p


def jp_compress_accurate(input_image_p, target_bpp, verbose=False):
	out_path = os.path.splitext(input_image_p)[0] + '_out_jp.jpeg'
	img = Image.open(input_image_p)
	dim = float(img.size[0] * img.size[1])
	for q in range(0, 99):
	# for q in np.arange(1, 99, 0.1):
		img.save(out_path, quality=q, subsampling=0) # 4:2:2
		bpp = (8 * _jpeg_content_length(out_path)) / dim
		if bpp > target_bpp:
			if verbose:
				print('q={} -> {}bpp'.format(q, bpp))
			return out_path, bpp
	raise ValueError(
		'Cannot achieve target bpp {} with JPEG for image {} (max {})'.format(target_bpp, input_image_p, bpp))


def _jpeg_content_length(p):
	"""
	Determines the length of the content of the JPEG file stored at `p` in bytes, i.e., size of the file without the
	header. Note: Note sure if this works for all JPEGs...
	:param p: path to a JPEG file
	:return: length of content
	"""
	with open(p, 'rb') as f:
		last_byte = ''
		header_end_i = None
		for i in itertools.count():
			current_byte = f.read(1)
			if current_byte == b'':
				break
			# some files somehow contain multiple FF DA sequences, don't know what that means
			if header_end_i is None and last_byte == b'\xff' and current_byte == b'\xda':
				header_end_i = i
			last_byte = current_byte
		# at this point, i is equal to the size of the file
		return i - header_end_i - 2  # minus 2 because all JPEG files end in FF D0


def jp_measure_over_interval(input_image_p, fout, q_grid):
	input_image_root_p, ext = os.path.splitext(input_image_p)
	assert ext == '.png'
	for q in q_grid:
		actual_bpp, output_image_jp_p = jp_compress(input_image_p, q)
		im_in = scipy.ndimage.imread(input_image_p)
		im_out = scipy.ndimage.imread(output_image_jp_p)
		ssim, msssim, psnr = compare_imgs.compare(
			im_in, im_out, calc_ssim=True, calc_msssim=True, calc_psnr=True)
		fout_str = _append_to_measures_f(fout, q, actual_bpp, ssim, msssim, psnr)
		print(fout_str, end='\r')
		os.remove(output_image_jp_p)
	print("-------------------------")


# ---------------------ABOUT JP2k----------------------------------
def jp2k_compress(input_image_p, target_bpp, no_weights=True):
	output_image_j2_p = os.path.splitext(input_image_p)[0] + '_out_jp2.jp2'
	# kdu can only work with "tif", "tiff", "bmp", "pgm", "ppm", "raw" and "rawl"
	with remove_file_after(convert_im_to('bmp', input_image_p)) as input_image_bmp_p:
		cmd = [KDU_COMPRESS,
			   '-i', input_image_bmp_p, '-o', output_image_j2_p,
			   '-rate', str(target_bpp), '-no_weights']
		output = subprocess.check_output(cmd).decode()
		actual_bpp = float(re.search(_KDU_RE_PAT, output).group(1))
		return output_image_j2_p, actual_bpp


def jp2k_compress_accurate(input_image_p, target_bpp, verbose=False, delta=0.005):
	for i in range(25):
		out_path, actual_bpp = jp2k_compress(input_image_p, target_bpp + i * delta)
		if actual_bpp >= target_bpp:
			if verbose:
				print('target={} -> actual={}bpp'.format(target_bpp, actual_bpp))
			return out_path, actual_bpp
	raise ValueError(
		'Cannot achieve target bpp {} with JPEG2K for image {} (max {}bpp)'.format(
			target_bpp, input_image_p, actual_bpp))


def jp2k_measure_over_interval(input_image_p, fout, bpp_grid):
	input_image_root_p, ext = os.path.splitext(input_image_p)
	assert ext == '.png', 'Expected PNG to convert to BMP, got {}'.format(input_image_p)

	for bpp in bpp_grid:
		output_image_j2_p, actual_bpp = jp2k_compress(input_image_p, bpp)
		im_in = scipy.ndimage.imread(input_image_p)
		# im_in = plt.imread(input_image_p)
		# im_out = plt.imread(output_image_j2_p)
		im_out = scipy.ndimage.imread(output_image_j2_p)
		ssim, msssim, psnr = compare_imgs.compare(
			im_in, im_out, calc_ssim=True, calc_msssim=True, calc_psnr=True)
		fout_str = _append_to_measures_f(fout, bpp, actual_bpp, ssim, msssim, psnr)
		# print('bpp', 'actual_bpp', 'ssim', 'msssim', 'psnr')
		print(fout_str, end='\r')
		os.remove(output_image_j2_p)
	print()


# -----------------------ABOUT webp--------------------------------
def _webp_compress(input_image_p, q):
	"""
	:param input_image_p:
	:param q:
	:return: out_p, bpp
	"""
	output_image_webp_p = input_image_p.replace('.png', '_tmp_webp.webp')

	cmd = [CWEBP, '-q', str(q), input_image_p, '-o', output_image_webp_p]
	# cwebp writes to stdout which makes this a pain
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
	cwebp_output, _ = process.communicate()
	bpp = _parse_webp_output(cwebp_output)
	return output_image_webp_p, bpp


def _decode_webp_to_png(webp_p):
	png_p = webp_p.replace('.webp', '_as_png.png')
	subprocess.call([DWEBP, webp_p, '-o', png_p], stderr=subprocess.DEVNULL)
	return png_p


def _parse_webp_output(otp):
	"""
	Sample output:
		Saving file 'test.webp'
		File:      /home/mentzerf/phd_code/data/kodak/kodim03.png
		Dimension: 768 x 512
		Output:    13656 bytes Y-U-V-All-PSNR 36.19 41.08 42.41   37.38 dB
		block count:  intra4:        822  (53.52%)
					  intra16:       714  (46.48%)
					  skipped:       127  (8.27%)
		bytes used:  header:            104  (0.8%)
					 mode-partition:   3337  (24.4%)
		 Residuals bytes  |segment 1|segment 2|segment 3|segment 4|  total
			macroblocks:  |       5%|      10%|      27%|      58%|    1536
			  quantizer:  |      64 |      62 |      53 |      41 |
		   filter level:  |      22 |      15 |      55 |      43 |

	:param otp:
	:return:
	"""
	w, h = _match_regex_ungroup_as_int(otp, r'Dimension: (\d+) x (\d+)')
	total_bytes = _match_regex_ungroup_as_int(otp, r'Output:\s+(\d+)\s+bytes Y-U-V-All-PSNR')
	header_bytes = _match_regex_ungroup_as_int(otp, r'bytes used:\s+header:\s+(\d+)')
	content_bytes = total_bytes - header_bytes
	bpp = content_bytes * 8 / float(w * h)
	return bpp


def webp_compress_accurate(input_image_p, target_bpp, verbose=False):
	# for q in range(0, 101):
	for q in np.arange(0, 101, 0.01):
		out_p, actual_bpp = _webp_compress(input_image_p, q)
		if actual_bpp >= target_bpp:
			if verbose:
				print('q={} -> {}bpp'.format(q, actual_bpp))
			return out_p, actual_bpp
	raise ValueError(
		'Cannot achieve target bpp {} with webp for image {} (max {})'.format(target_bpp, input_image_p, actual_bpp))


def _match_regex_ungroup_as_int(s, r):
	m = re.search(r, s)
	if not m:
		raise ValueError('Did not match regex {} in {}'.format(r, s))
	otp = tuple(map(int, m.groups()))
	return otp[0] if len(otp) == 1 else otp


def webp_measure_over_interval(input_image_p, fout, q_grid):
	input_image_root_p, ext = os.path.splitext(input_image_p)
	assert ext == '.png'

	for q in map(int, q_grid):
		out_p, bpp = _webp_compress(input_image_p, q)
		with remove_file_after(out_p) as out_p:
			with remove_file_after(_decode_webp_to_png(out_p)) as out_png_p:
				im_in = scipy.ndimage.imread(input_image_p)
				im_out = scipy.ndimage.imread(out_png_p)
				ssim, msssim, psnr = compare_imgs.compare(
					im_in, im_out, calc_ssim=True, calc_msssim=True, calc_psnr=True)
				fout_str = _append_to_measures_f(fout, q, bpp, ssim, msssim, psnr)
				print(fout_str, end='\r')
	print()


# -----------------------------------------------------------------
# high level API
def plot_measured_dataset(bpg_root_dir, out_dir):
   for p in all_measures_file_ps(bpg_root_dir):
       print(p)
       plt.figure(figsize=(5, 5))
       # bpps, msssims = ft.unzip(read_measures(p))
       bpps, msssims = zip(*read_measures(p, metric='ms-ssim'))
       plt.plot(bpps, msssims)
       plt.title(os.path.basename(p))
       plt.xlim((0, 1.5))
       plt.ylim((0.4, 1))
       out_p = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '.pdf')
       print(out_p, end='\r')
       plt.savefig(out_p, bbox_inches='tight')
       plt.close()


def gen_bpg(in_imgs, out_dir, qs, first_n):
	"""
	generate compressed bpg img, only for bpg codec
	:param in_imgs: input img dir
	:param out_dir: output dir
	:param qs: img quality, list
	:param first_n: integer, int, for first-n output
	:return: NOne
	"""
	if '*' not in in_imgs:
		in_imgs = os.path.join(in_imgs, '*.png')
	images = sorted(glob.glob(in_imgs))[:first_n]
	assert len(images) > 0, 'No matches for {}'.format(in_images)
	for img in images:
		if 'tmp' in img:
			print('Skipping {}'.format(img))
			continue
		shutil.copy(img, os.path.join(out_dir, os.path.basename(img).replace('.png', '_base.png')))
		print(os.path.basename(img))
		for q in qs:
			with remove_file_after(bpg_compress(img, q=q, tmp_dir=out_dir, chroma_fmt='422')) as p:
				bpp = bpp_of_bpg_image(p)
				out_png = decode_bpg_to_png(p)
				out_name = os.path.basename(img).replace('.png', '_{:.4f}.png'.format(bpp))
				out_p = os.path.join(out_dir, out_name)
				print('-> {:.3f}: {}'.format(bpp, out_name))
				os.rename(out_png, out_p)
		# remove base.png
		os.remove(os.path.join(out_dir, os.path.basename(img).replace('.png', '_base.png')))


def compress_to_bpp(root_dir, out_dir, target_bpp, mode):
	"""
	compress image by mode, up to target_bpp
	:param root_dir:
	:param out_dir:
	:param target_bpp: float
	:param mode: float, bpg, jp2k, jp, webp
	:return:
	"""

	def target_p(img_, bpp_):
		return os.path.join(
			out_dir,
			os.path.splitext(os.path.basename(img_))[0] +
			'_{}_{:.5f}.png'.format(mode, bpp_)
		)

	for img in _get_image_paths(root_dir):
		if mode == 'bpg':
			bpg_p, actual_bpp = _bpg_compress_to_bpp(img, target_bpp, precise=True, tmp_dir=out_dir)
			with remove_file_after(bpg_p):
				png_p = decode_bpg_to_png(bpg_p)
		elif mode == 'jp2k':
			jp2k_p, actual_bpp = jp2k_compress_accurate(img, target_bpp, verbose=True)
			with remove_file_after(jp2k_p):
				png_p = convert_im_to('.png', jp2k_p)
		elif mode == 'jp':
			jp_p, actual_bpp = jp_compress_accurate(img, target_bpp, verbose=True)
			with remove_file_after(jp_p):
				png_p = convert_im_to('.png', jp_p)
		elif mode == 'webp':
			webp_p, actual_bpp = webp_compress_accurate(img, target_bpp, verbose=True)
			with remove_file_after(webp_p):
				png_p = _decode_webp_to_png(webp_p)
		else:
			raise ValueError('Invalid mode {}'.format(mode))
		print('{} -> {:.3f}bpp (target: {:.3f} bpp)'.format(
			img, actual_bpp, target_bpp))
		shutil.move(png_p, target_p(img, actual_bpp))  # rename file


def create_curves_for_images(root_dir, out_dir, grid, mode):
	os.makedirs(out_dir, exist_ok=True)
	times = []
	all_img_ps = _get_image_paths(root_dir)
	assert len(all_img_ps) > 0
	measure_over_interval = {
		'bpg': bpg_measure_over_interval,
		'jp2k': jp2k_measure_over_interval,
		'jp': jp_measure_over_interval,
		'webp': webp_measure_over_interval
	}[mode]

	for i, img_p in enumerate(all_img_ps):
		if 'tmp' in img_p:
			print('Skipping {}...'.format(img_p))
			continue
		img_name = os.path.splitext(os.path.basename(img_p))[0]
		print(img_name)
		s = time.time()
		mf = measures_file_p(out_dir, img_name)
		if os.path.exists(mf):
			continue
		with open(mf, 'w+') as f:
			measure_over_interval(img_p, f, grid)
		times.append(time.time() - s)
		avg_time = np.mean(times[-15:])
		print('Time left: {:.2f}min'.format(avg_time * (len(all_img_ps) - i) / 60))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('root_dir', help='dir of imgs')
	parser.add_argument('out_dir', help='where to save')
	parser.add_argument('--modes', type=str, choices=['all', 'bpg', 'jp2k', 'jp', 'webp'], nargs='+')
	parser.add_argument('--plot', action='store_const', const=True, help='if Given, plot;else no')
	parser.add_argument('--bpp', type=float,
						help='if given, try to compress imgs to given bpp.')
	parser.add_argument('--gen_q', type=int, nargs='+',
						help='only generate output for specific q.Only support "modes == bpg"')
	parser.add_argument('--first_n', type=int, metavar='N',
						help='If given with --gen_q: only generate output for first N pics.')
	parser.add_argument('--grid', type=float, nargs='+', help='Generate output for multiple q.')
	flags = parser.parse_args()

	if flags.modes == ['all']:
		flags.modes = ['bpg', 'jp2k', 'jp', 'webp']

	print("file will be saved in {}".format(flags.out_dir))
	os.makedirs(flags.out_dir, exist_ok=True)

	if 'jp2k' in flags.modes:
		check_if_program_is_available(KDU_COMPRESS, 'kdu_compress', 'KDU_COMPRESS')
	if 'webp' in flags.modes:
		check_if_program_is_available(CWEBP, 'cwebp', 'CWEBP')
		check_if_program_is_available(DWEBP, 'dwebp', 'DWEBP')
	if 'bpg' in flags.modes:
		check_if_program_is_available(BPGENC, 'bpgenc', 'BPGENC')

	# compress
	if flags.gen_q:
		assert flags.modes == ['bpg']
		gen_bpg(flags.root_dir, flags.out_dir, flags.gen_q, flags.first_n)
	elif flags.bpp:
		for mode in flags.modes:
			compress_to_bpp(flags.root_dir, flags.out_dir, target_bpp=flags.bpp, mode=mode)
	else:
		# TODO: modify
		if flags.plot:
			# raise NotImplementedError()
			plot_measured_dataset(flags.root_dir, flags.out_dir)
		else:
			for mode in flags.modes:
				if not flags.grid:
					grids = {
						'bpg': [5, 20, 30, 33, 36, 40, 43, 46, 50],  # q
						'webp': [0, 2, 4, 8, 15, 25, 40, 60, 80, 100],  # q
						'jp2k': [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.2, 1.4, 1.6],  # bpp
						'jp': [1, 3, 4, 5, 10, 15, 25, 35, 45, 60, 87, 90, 95, 98]  # bpp
					}
					flags.grid = grids[mode]
				out_dir = os.path.join(flags.out_dir, "out_{}".format(mode))
				create_curves_for_images(flags.root_dir, out_dir, flags.grid, mode)
				grids = None



if __name__ == "__main__":
	main()
