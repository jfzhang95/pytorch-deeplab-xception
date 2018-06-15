class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/zvision/Databases/Segmentation/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return '/home/zvision/Databases/Segmentation/benchmark_RELEASE/' # folder that contains dataset/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
