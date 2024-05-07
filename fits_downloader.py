import alminer

# * Hämtar sökresultat från ALMA med keywords: "Disks around high-mass stars" och/eller "Disks around low-mass stars"
# observations = alminer.keysearch({'science_keyword':["'Disks around high-mass stars'","'Disks around low-mass stars'"]}, print_targets=False)
# * Keysearch ovan returnerar ca 25000 resultat. 
# observations = alminer.keysearch({'proposal_id':["2016.1.00484.L"], 'science_keyword':["'Disks around high-mass stars'","'Disks around low-mass stars'"]}, print_targets=False)
# observations = alminer.keysearch({'proposal_id':["2016.1.00484.L"]}, print_targets=False)

# print(observations.shape)

def download_Dsharp():
    observations = alminer.keysearch({'proposal_id':["2016.1.00484.L"]}, print_targets=False)

    print(observations.shape)

    # * Denna loop laddar ner de första 1500 resultate i klumpar om 50.
    for i in range(100,140,10):
        print(i,"->",i+10)
        selected = observations.iloc[i:i+10]
        alminer.download_data(selected, fitsonly=True, dryrun=False, location='./data/D-sharp', filename_must_include=['_sci', '.pbcor', 'cont'], print_urls=False)


def download_pbcor():
    start_at = 1300
    amount = 700
    log_step = 50
    
    observations = alminer.keysearch({'science_keyword':["'Disks around high-mass stars'","'Disks around low-mass stars'"]}, print_targets=False)

    print(observations.shape)
    # * Denna loop laddar ner de första 1500 resultate i klumpar om 50.
    for i in range(start_at,start_at+amount,log_step):
        print(i,"->",i+log_step)
        selected = observations.iloc[i:i+log_step]
        alminer.download_data(selected, fitsonly=True, dryrun=False, location='./data/fits_tracked', filename_must_include=['_sci', '.pbcor', 'cont'], print_urls=True)

def download_pb():
    start_at = 20000
    amount = 5000
    log_step = 50
    
    observations = alminer.keysearch({'science_keyword':["'Disks around high-mass stars'","'Disks around low-mass stars'"]}, print_targets=False)

    print(observations.shape)
    # * Denna loop laddar ner de första 1500 resultate i klumpar om 50.
    for i in range(start_at,start_at+amount,log_step):
        print(i,"->",i+log_step)
        selected = observations.iloc[i:i+log_step]
        # print("--- .pbcor ---")
        # alminer.download_data(selected, fitsonly=True, dryrun=False, location='./data/fits/pb', filename_must_include=['_sci', '.pbcor', 'cont'], print_urls=False)
        print("--- .pb ---")
        alminer.download_data(selected, fitsonly=True, dryrun=False, location='./data/fits/pb', filename_must_include=['_sci','.pb.', 'cont'], print_urls=False)
        # print("--- .flux ---")
        # alminer.download_data(selected, fitsonly=True, dryrun=False, location='./data/fits/pb', filename_must_include=['_sci', '.flux', 'cont'], print_urls=False)

if __name__ == "__main__":
    download_pbcor()