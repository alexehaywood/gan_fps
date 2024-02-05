
setwd("~/Documents/Work/GAN_Paper/Code/GAN_Data/Real_example")

library(tidyverse)
library(GEOquery)

library(foreach)
library(doParallel) 


##Define function to get expression data from GEO database and format it
write_csv(tibble(c(1,2)), 'test.csv')

#gsm <- gsm_ids[1]
get_rand_sample <- function(gsm, features, n_noFeature){
  ##############################################################################
  ##Get sample from GEO database by gsm identifier
  #gsm: type character of sample gsm identifier 
  #features: the genes required in the sample data
  #n_noFeatures: how many missing genes from feature list to allow for
  ##############################################################################
  
  #get sample information from GEO
  dat_gsm <- getGEO(gsm, getGPL = TRUE)
  dat_exp <- Table(dat_gsm)
  if(dim(dat_exp)[1] == 0){ #ensure expression data is present
    return(NULL)
  }
  #platform data for the sample from GEO (used to get gene names)
  gpl <- Meta(dat_gsm)$platform_id
  label <- Meta(dat_gsm)$characteristics_ch1[1]
  dat_gpl <- Table(getGEO(gpl))
  
  #get gene names from reference IDs if present, otherwise return NULL
  col_gene_symbol_id <- c('orf', 'gene symbol', 'gene_symbol', 'ilmn_gene', 'genesymbol') #where gene symbols located
  cols_keep <- c(1, which(tolower(colnames(dat_gpl)) %in% col_gene_symbol_id)) #convert all to lowercase to make matching columns more efficient
  if (length(cols_keep) > 1){ #check if a gene names column was found
    dat_gpl <- dat_gpl %>% select(all_of(cols_keep))
  }
  else{
    return(NULL)
  }
  #map reference IDs to gene names
  dat_exp[1] <- plyr::mapvalues(as.vector(dat_exp[1])[[1]], from = as.vector(dat_gpl[1])[[1]], to =
                                  as.vector(dat_gpl[2])[[1]], warn_missing = FALSE) 
  
  #check how many genes are missing from feature list inputted, and return NULL if too many
  if (length(features) - length(which( features %in% tolower(as.vector(dat_exp[1])[[1]]) )) > n_noFeature){
    return(NULL)
  }
  
  #subset data for required features only
  dat_exp <- dat_exp[which( tolower(as.vector(dat_exp[1])[[1]]) %in% features ) ,]
  dat_exp <- dat_exp[!duplicated(dat_exp[1]), ] #take the 1st of each gene entry (I guess isoforms lead to dups?)
  
  dat_exp <- rbind(c('label', label), dat_exp)
  
  #"un-transform" all log2 count data (ensure all data are transformed the same) 
  #preproc_id <- c('log2', 'log 2', '2 logarithm', 'rma' ,'genespring', 'agilent')
  #preproc_info <- tolower(paste(dat_gsm@header[["data_processing"]], collapse = ' '))
  #pre_proc_found <- unlist(lapply(preproc_id, 
  #                                function(x) grepl( x, preproc_info, fixed = TRUE)
  #                                ))
  #if (sum(pre_proc_found) > 0){
  #  dat_exp[,2] <- 2^dat_exp[,2]
  #}
  
  colnames(dat_exp)[2] <- gsm #include sample identifier in output
  return(dat_exp[1:2])

}

process_out <- function(x){
  ##############################################################################
  ##Process the output from get_rand_sample()
  #x: output from get_rand_sample()
  ##############################################################################
  if (!is.null(x)){
    x <- t(x)
    colnames(x) <- tolower(x[1,])
    id <- rownames(x)[2]
    x <- x[2, ]
    x <- c(id = id, x)
    
    return( base::as.data.frame(t(x)) )
  }
  else{
    return(NULL)
  }
}



#Get the GSM values from the outputted file above

acc_ext <- 'Accessions_GSE14520.txt'
acc_val <- 'Accessions_GSE25097.txt'
acc_real <- 'Accessions_GSE36376.txt'

#get accession codes
for(j in seq(1, 3, 1)){
  if(j == 1){
    acc <- acc_val
  }
  if(j == 2){
    acc <- acc_ext
  }
  if(j == 3){
    acc <- acc_real
  }

  
  dat_acc <- t(read_tsv(acc, col_names = FALSE))
  acc_unique <- unique(dat_acc[, 1])
  
  #Define feature to extract from GEO samples, and how many GSM values to pass into the get_rand_sample() function
  dge_real <- read_csv('dge_real.csv')
  features <- dge_real %>% arrange(adj.P.Val) %>% select(1) %>% unlist() %>% unname() %>% tolower()
  features <- features[1:200]
  
  gsm_ids <- sample(dat_acc[, 1], length(dat_acc[, 1]))
  #gsm_ids <- gsm_ids[!grepl(paste0(dat_ext$id[1:length(dat_ext$id)], collapse = "|"), gsm_ids)]
  
  #Used parallel processing to create dataset, to improve computation time
  numCores <- detectCores()
  registerDoParallel(numCores - 3)
  dat_get <- foreach(i = seq(1, length(gsm_ids)), .combine = rbind, .errorhandling = 'remove') %dopar% {
    gc(verbose = FALSE)
    gsm <- gsm_ids[i]
    x <- get_rand_sample(gsm, features, 100) %>% process_out()
    
    dat_new <- base::data.frame(matrix(ncol = length(features)+2, nrow = 0))
    colnames(dat_new) <- c('id', 'label', features)
    
    merge(dat_new, x, all = TRUE)[, c('id', 'label', features)]
  }
  
  if(acc == acc_real){
    write_csv(dat_get, 'dat_real_combo.csv')
  }
  if(acc == acc_ext){
    write.csv(dat_get %>% select(-2), 'dat_ext.csv')
  }
  if(acc == acc_val){
    write_csv(dat_get, 'dat_val.csv')
  }

}


#Get which genes to use
dat_real <- read_csv('dat_real_combo.csv', col_types = paste('cf', paste(rep('d', 200), collapse = ''), sep = '') )
dat_ext <- read_csv('dat_ext.csv', col_types = paste('cf', paste(rep('d', 200), collapse = ''), sep = '') )
dat_val <- read_csv('dat_val.csv', col_types = paste('cf', paste(rep('d', 200), collapse = ''), sep = '') )

##check all na values are indeed na, not a string or something
##check igfals from the ext dat (an na got through here)
missing_real <- dat_real %>% select(-1, -2) %>% summarise(across(.cols = everything(), .fns = ~sum(is.na(.x)))) %>% t() %>% as.data.frame()
missing_ext <- dat_ext %>% select(-1, -2) %>% summarise(across(.cols = everything(), .fns = ~sum(is.na(.x)))) %>% t() %>% as.data.frame()
missing_val <- dat_val %>% select(-1, -2) %>% summarise(across(.cols = everything(), .fns = ~sum(is.na(.x)))) %>% t() %>% as.data.frame()

##test 
missing_ext['igfals',]
sum(is.na(dat_ext[, 'igfals']))
##No NA in igfals

missing_summary <- cbind(rownames(missing_real), missing_real, missing_ext, missing_val)
colnames(missing_summary) <- c('gene', 'real', 'ext', 'val')
missing_summary <- missing_summary %>% rowwise() %>% mutate(sum = sum(real+ext+val)) %>% select(gene, sum) %>% arrange(desc(sum))
print(missing_summary)


id_remove <- c(missing_summary[seq(1, 65), 1])$gene
print(paste(id_remove, collapse = ', '))
print(length(id_remove))


#label names
print(levels(dat_real$label))
print(levels(dat_val$label))

#tidy
dat_real <- dat_real %>% select(-all_of(id_remove)) %>% filter(label == 'tissue: liver tumor' | label == 'tissue: adjacent non-tumor liver')
dat_val <- dat_val %>% select(-all_of(id_remove)) %>% filter(label == "tissue: tumor liver" | label == "tissue: non_tumor liver")
dat_ext <- dat_ext %>% select(-all_of(id_remove)) %>% select(-1)

#change label names
print(levels(dat_real$label))
print(levels(dat_val$label))

##need to explicilty state which labels == which
levels(dat_real$label) <- plyr::revalue(dat_real$label, c('tissue: liver tumor' = 'case', 'tissue: adjacent non-tumor liver' = 'control'))
levels(dat_val$label) <- plyr::revalue(dat_val$label, c("tissue: non_tumor liver" = 'control',
                                                         "tissue: tumor liver" = 'case',
                                                         "tissue: cirrhotic liver" = 'NA',
                                                         "tissue: healthy liver" = 'NA2'))



write.csv(dat_real, 'dat_real_combo.csv')
write.csv(dat_val, 'dat_val.csv')
write.csv(dat_ext, 'dat_ext.csv')



#Get data ready for each experiment
dat_combo <- read_csv('dat_real_combo.csv', col_types = paste('ccf', paste(rep('d', 200), collapse = ''), sep = '') ) %>% select(-1)

summary(dat_combo$label)

icesTAF::mkdir(c('1a', '1b', '1c',
                 '2a', '2b', '2c',
                 '3a', '3b', '3c',
                 '4a', '4b', '4c',
                 '5a', '5b', '5c',
                 '6a', '6b', '6c', 
                 '7a', '7b', '7c', 
                 '8a', '8b', '8c', 
                 '9a', '9b', '9c'))

for(i in seq(1, 9, 1)){
  dir1 = paste(as.character(i), 'a', sep = '')
  dir2 = paste(as.character(i), 'b', sep = '')
  dir3 = paste(as.character(i), 'c', sep = '')
  
  if(i == 1){
    class_imbal <- 0.4
    n_control <- 40
  }
  if(i == 2){
    class_imbal <- 0.4
    n_control <- 80
  }
  if(i == 3){
    class_imbal <- 0.4
    n_control <- 120
  }
  if(i == 4){
    class_imbal <- 0.5
    n_control <- 40
  }
  if(i == 5){
    class_imbal <- 0.5
    n_control <- 80
  }
  if(i == 6){
    class_imbal <- 0.5
    n_control <- 120
  }
  if(i == 7){
    class_imbal <- 0.6
    n_control <- 40
  }
  if(i == 8){
    class_imbal <- 0.6
    n_control <- 80
  }
  if(i == 9){
    class_imbal <- 0.6
    n_control <- 120
  }
  
  set.seed(200)
  dat_under <- dat_combo %>% filter(label == 'case') %>% sample_n( n_control*class_imbal )
  dat_over <- dat_combo %>% filter(label == 'control') %>% sample_n( n_control )
  
  write_csv(dat_under, paste(dir1, 'dat_real_class1.csv', sep = '/'))
  write_csv(dat_over, paste(dir1, 'dat_real_class2.csv', sep = '/'))
  dat_under %>% add_row(dat_over) %>% write_csv(paste(dir1, 'dat_real_combo.csv', sep = '/'))
  
  write_csv(dat_under, paste(dir2, 'dat_real_class1.csv', sep = '/'))
  write_csv(dat_over, paste(dir2, 'dat_real_class2.csv', sep = '/'))
  dat_under %>% add_row(dat_over) %>% write_csv(paste(dir2, 'dat_real_combo.csv', sep = '/'))
  
  
  write_csv(dat_under, paste(dir3, 'dat_real_class1.csv', sep = '/'))
  write_csv(dat_over, paste(dir3, 'dat_real_class2.csv', sep = '/'))
  dat_under %>% add_row(dat_over) %>% write_csv(paste(dir3, 'dat_real_combo.csv', sep = '/'))
  
} 




#Below used for testing specific IDs
#gsm <- gsm_ids[40]
#dat_gsm <- getGEO('GSM890272', getGPL = TRUE)
#dat_gsm
