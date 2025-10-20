import os
import sys
from collections import OrderedDict


def _validate_input(inputfile, outputfile):
    # Python 2 compatibility
    if sys.version_info.major == 3:
        import io
        basestring = str
        file = io.TextIOBase

    if isinstance(inputfile, basestring):
        # Pre-load the file in memory, if vmtouch is installed.
        os.system('vmtouch -qt {} & >/dev/null 2>/dev/null'.format(inputfile))

        input_ = open(inputfile)
        close_input = True

    elif isinstance(inputfile, file):
        input_ = inputfile
        close_input = False

        if not input_.read(1):
            raise IOError('The input file appears empty')
        input_.seek(-1)

    else:
        raise ValueError('inputfile should be a file name or a'
                         'file handler, got {} instead.'.format(
            type(inputfile)))

    if isinstance(outputfile, basestring):
        output = open(outputfile, 'w')
        close_output = True
    elif isinstance(outputfile, file):
        output = outputfile
        close_output = False
    else:
        raise ValueError('outputfile should be a file name or a'
                         'file handler, got {} instead.'.format(
            type(outputfile)))

    return input_, close_input, output, close_output


def _read_sto(inputfile):
    data = OrderedDict()
    reference_seq = []

    # Skip header
    for line in inputfile:
        line = line.strip()
        if line and not line.startswith('#'):
            header, sequence = line.split()
            break
    else:
        raise IOError('The file appears to be empty')

    original_header = header

    while True:
        assert header == original_header, (header, original_header)
        reference_seq.extend((s for s in sequence if s != '-'))
        index = [True if s != '-' else False for s in sequence]

        for line in inputfile:
            line = line.strip()
            if not line:
                # Ignore empty lines
                continue

            if line == '//':
                return header, reference_seq, data

            if not line.startswith('#'):
                name, sequence = line.split()
                if name == original_header:
                    break

                # Fist iteration, a new protein:
                if not name in data:
                    data[name] = []


                data[name].extend(s if i else s.lower()
                                  for s, i in zip(sequence, index)
                                  if i or s != '-')

        assert header == original_header, (header, original_header)




def parse_a3m(inputfile, outputfile):
    input_, close_input, output, close_output = _validate_input(inputfile,
                                                                outputfile)

    header, reference_seq, data = _read_sto(input_)

    if close_input:
        input_.close()
    os.system('vmtouch -qe {} & >/dev/null 2>/dev/null'.format(inputfile))

    # Write to file
    output.write(''.join(('>', header, '\n')))
    output.write(''.join(reference_seq))
    output.write('\n')

    for name, seq in data.items():
        output.write(''.join(('>', name, '\n')))
        output.write(''.join(seq))
        output.write('\n')

    if close_output:
        output.close()
    else:
        output.flush()


def parse_fasta(inputfile, outputfile):
    input_, close_input, output, close_output = _validate_input(inputfile,
                                                                outputfile)

    header, reference_seq, data = _read_sto(input_)

    if close_input:
        input_.close()
    os.system('vmtouch -qe {} & >/dev/null 2>/dev/null'.format(inputfile))

    # Write to file
    output.write(''.join(('>', header, '\n')))
    output.write(''.join(reference_seq))
    output.write('\n')

    for name, seq in data.items():
        output.write(''.join(('>', name, '\n')))
        output.write(''.join((s for s in seq if not s.islower())))
        output.write('\n')

    if close_output:
        output.close()
    else:
        output.flush()


def parse_aln(inputfile, outputfile):
    # This is as parse_fasta, but without printing the headers in the output.
    input_, close_input, output, close_output = _validate_input(inputfile,
                                                                outputfile)

    header, reference_seq, data = _read_sto(input_)

    if close_input:
        input_.close()
    os.system('vmtouch -qe {} & >/dev/null 2>/dev/null'.format(inputfile))

    # Write to file
    output.write(''.join(reference_seq))
    output.write('\n')

    for name, seq in data.items():
        output.write(''.join((s for s in seq if not s.islower())))
        output.write('\n')

    if close_output:
        output.close()
    else:
        output.flush()


if __name__ == '__main__':
    f='/home/david/Dropbox/arne_lab/alignments_parsed/H9999.jhE0.sto'
    parse_aln(f, f.replace('.sto', '.aln'))
