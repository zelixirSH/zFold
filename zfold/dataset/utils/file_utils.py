"""File-related utility functions."""

import os
import re
import shutil
import tempfile


def get_tmp_dpath():
    """Get the directory path to temporary files.

    Args: n/a

    Returns:
    * tmp_dpath: directory path to temporary files
    """

    assert 'TMPDIR_PRFX' in os.environ, 'environmental variable <TMPDIR_PRFX> not defined'
    tmp_dpath_prfx = os.getenv('TMPDIR_PRFX')
    os.makedirs(os.path.dirname(tmp_dpath_prfx), exist_ok=True)
    tmp_dpath = tempfile.TemporaryDirectory(prefix=tmp_dpath_prfx).name
    os.makedirs(tmp_dpath, exist_ok=True)

    return tmp_dpath


def clear_tmp_files():
    """Clear-up temporary files.

    Args: n/a

    Returns: n/a
    """

    assert 'TMPDIR_PRFX' in os.environ, 'environmental variable <TMPDIR_PRFX> not defined'
    tmp_dpath_prfx = os.getenv('TMPDIR_PRFX')
    root_dir = os.path.dirname(tmp_dpath_prfx)
    tmp_dname_prfx = os.path.basename(tmp_dpath_prfx)
    for name in os.listdir(root_dir):
        if name.startswith(tmp_dname_prfx):
            tmp_dpath = os.path.join(root_dir, name)
            shutil.rmtree(tmp_dpath)


def find_files_by_suffix(root_dir, suffix):
    """Recursively find all the files with certain suffix (e.g. '.pdb').

    Args:
    * root_dir: root directory path
    * suffix: file name suffix

    Returns:
    * file_paths: full paths to all the matched files
    """

    file_paths = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path):
            file_paths.extend(find_files_by_suffix(full_path, suffix))
        elif name.endswith(suffix):
            file_paths.append(full_path)

    return file_paths


def recreate_directory(path):
    """Re-create the directory.

    Args:
    * path: path to the directory to be re-created

    Returns: n/a
    """

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def unpack_archive(axv_fpath, root_dir):
    """Unpack a *.tar.gz / *.tgz / *.tar archive file to the specified directory.

    Args:
    * axv_fpath: path to the *.tar.gz / *.tgz / *.tar archive file
    * root_dir: directory path to unpack the archive file

    Returns:
    * axv_dpath: directory path to the unpacked archive file
    """

    regex = re.compile(r'\.(tar\.gz|tgz|tar)$')
    axv_fname = os.path.basename(axv_fpath)
    suffix = re.search(regex, axv_fname).group()
    axv_dname = axv_fname.replace(suffix, '')
    axv_dpath = os.path.join(root_dir, axv_dname)
    if os.path.exists(axv_dpath):
        shutil.rmtree(axv_dpath)
    os.makedirs(root_dir, exist_ok=True)
    shutil.unpack_archive(axv_fpath, root_dir)

    return axv_dpath


def make_archive(axv_dpath, axv_fpath):
    """Pack a directory into a *.tar.gz / *.tgz / *.tar archive file.

    Args:
    * axv_dpath: directory path to the unpacked archive file
    * axv_fpath: path to the *.tar.gz / *.tgz / *.tar archive file

    Return: n/a
    """

    regex = re.compile(r'\.(tar\.gz|tgz|tar)$')
    base_name = re.sub(regex, '', axv_fpath)
    axv_format = 'tar' if axv_fpath.endswith('.tar') else 'gztar'
    root_dir = os.path.dirname(axv_dpath)
    base_dir = os.path.basename(axv_dpath)
    axv_fpath_tmp = shutil.make_archive(base_name, axv_format, root_dir, base_dir)
    if axv_fpath_tmp != axv_fpath:
        os.rename(axv_fpath_tmp, axv_fpath)
