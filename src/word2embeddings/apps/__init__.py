# -*- coding: utf-8 -*-
import sys


def use_theano_development_version():
    """Prepare usage of development version of Theano.

    Alters the PYTHONPATH variable by removing the paths to installed Theano
    versions and adding my own development installation.
    CAUTION: this function must be called before importing any of my or Theano's
    libraries.
    """
    print '\nold path:'
    print '\n'.join(sys.path)

    # List of possible Theano installation paths for different servers at CIS
    # and on my local machine.
    possible_paths = ['/usr/lib/python2.7/site-packages/Theano-0.6.0-py2.7.egg',
            'C:\\Anaconda\\lib\\site-packages\\theano-current',
            '/usr/local/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg',
            '/usr/lib/python2.7/site-packages/Theano-0.6.0rc3-py2.7.egg',
            '/usr/lib/python2.7/site-packages/Theano-0.6.0-py2.7.egg', #delta
            ]

    for p in possible_paths:

        try:
            sys.path.remove(p)
            print 'removed ', p
        except ValueError:
            pass

    sys.path.insert(0, '/mounts/Users/cisintern/ebert/data/promotion/src/theano/')
    sys.path.insert(0, 'Z:\\data\\promotion\\src\\theano\\')
    #sys.path.insert(0, '/mounts/Users/student/irina/Programs/Theano/Theano/')

    print 'new path:'
    print '\n'.join(sys.path)

    from theano import version

    print '\nnew Theano version:', version.full_version

#     sys.path.remove('/usr/lib/python2.7/site-packages')
