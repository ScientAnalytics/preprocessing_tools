import tools.load_hyper as load_hyper
from tools.continuum_correction import continuum_image
import os.path as path
import os
import glob
import numpy
from spectral import envi
import click


def folder_to_load(folder_location):
    print("Entered folder_to_load function...")
    files = glob.glob(folder_location + '*.hdr')
    print(f"Found {len(files)} .hdr files: {files}")  # Print the discovered files
    white_refs = list()
    dark_refs = list()
    images = list()
    to_load = list()

    for file in files:
        to_load.append(path.splitext(file)[0])

    for file in to_load:
        if 'whitereference' in file.lower():
            white_refs.append(file)
        elif 'darkreference' in file.lower():
            dark_refs.append(file)
        elif 'raw' in file.lower():
            images.append(file)
        else:
            print(f"Unrecognized file: {file}")
            continue
            raise FileNotFoundError(f'{file} is not a valid HSI')

    print(f"Identified {len(images)} raw images, {len(white_refs)} white references, and {len(dark_refs)} dark references.")
    print("Exiting folder_to_load function...")
    return (images, white_refs, dark_refs)


def white_dark_correction(hsi, white, dark):
    print("Entered white_dark_correction function...")
    white = white.interp(wavelength=hsi.wavelength)
    dark = dark.interp(wavelength=hsi.wavelength)
    bottom = (white - dark).squeeze('lines')
    top = hsi - dark.squeeze('lines')
    top_div_bottom = top / bottom
    print("Exiting white_dark_correction function...")
    return top_div_bottom


@click.command()
@click.option('--infolder', prompt='Folder to read',
              help='The location of the files you wish to load.')
@click.option('--outfolder', default=None,
              help='''Location of folder to write corrected images to.
              If not given, will be the same as the infolder.''')
@click.option('--force', default=False,
              help='Clobber existing files? Default is False.')
@click.option('--ignore_bands', default=10,
              help='Ignore the first n bands. Default is 10.')
@click.option('--drift', default=0,
              help='''Correct drift in wavelengths. Will add this value to the
              recorded wavelength values.''')
@click.option('--wd', default=False,
              help='''Perform white/dark reference correction. Will need a
              white and dark reference location to be supplied as well.''')
@click.option('--whiteref')
@click.option('--darkref')
def process_folder(infolder, outfolder, force, ignore_bands, drift,
                   wd, whiteref, darkref):
    print(f"Received infolder: {infolder}")
    print(f"Received outfolder: {outfolder}")

    print("Entered process_folder function...")
    print(f'Reading files in {infolder}.')
    if not outfolder:
        print('No outfolder supplied, using original folder.')
        outfolder = infolder
    elif not path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    # Load the good white and dark references once
    if wd:
        white_ref = load_hyper.load_image(good_white_ref_path)[:, :, ignore_bands:]
        dark_ref = load_hyper.load_image(good_dark_ref_path)[:, :, ignore_bands:]

    images = folder_to_load(infolder)[0]  # Assuming folder_to_load returns a tuple of lists

    for count, im in enumerate(images, start=1):
        print(f'''Processing file {path.basename(im)}
        {count} of {len(images)}''')
        hsi = load_hyper.load_image(im)[:, :, ignore_bands:]

        # Use hardcoded good white and dark references for correction
        fname = path.basename(hsi.attrs.get('filename')).replace('raw', 'wd-c-corr')
        fname = path.join(outfolder, fname)
        print(f'Force is set to {force}')
        if path.exists(fname) and not force:
            print(f'{fname} exists and "force" is false. No file written.')
            continue
        elif path.exists(fname) and force:
            print(f'{fname} exists and "force" is true. Will overwrite.')

        # White/Dark correction using good references
        print('White/Dark correction', end=' ')
        wd_corr = white_dark_correction(hsi, good_white_ref, good_dark_ref)
        print('done.')

        # Continuum correction
        print('Continuum correction', end=' ')
        c_corr = continuum_image(wd_corr)
        print('done.')

        c_corr.attrs = hsi.attrs
        print(f'Drift: {drift}')
        c_corr.attrs['wavelength'] = c_corr.wavelength.values + drift
        c_corr['wavelength'] = c_corr.wavelength.values + drift

        print(f'Saving file to {fname}.hdr', end=' ')
        envi.save_image(f'{fname}.hdr', c_corr.values,
                        dtype=numpy.float32,
                        interleave='bil', ext=None,
                        metadata=c_corr.attrs,
                        force=force)
        print('Done.')

    print("Exiting process_folder function...")


if __name__ == '__main__':
    process_folder()
