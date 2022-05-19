library(limma)
library(foreach)
library(doParallel)
require(svMisc)

wes = read.table("../../data/genomic/WES_variants_processed.csv.gz", stringsAsFactors = F, sep = ",", header = T, check.names = F)
protein = read.table("../../data/protein/protein_de_processed_rep.csv.gz", stringsAsFactors = F, sep = ",", header = T, row.names = 1, check.names = F)
meta = read.csv2("../../data/E0022_P06_final_sample_map_no_control.txt", sep = "\t", stringsAsFactors = F, check.names = F)

mut_counts = data.frame(cbind(colnames(wes[, -1]), colSums(wes[, -1])), stringsAsFactors = F)
colnames(mut_counts) = c("Gene", "mut_count")
genes = mut_counts[mut_counts$mut_count > 10, "Gene"]

protein = protein[, colnames(protein) %in% meta$Automatic_MS_filename]
genes = intersect(rownames(protein), genes)

# genes = c("TP53", "TTN")
cores = detectCores()
cl <- makeCluster(10) #not to overload your computer
registerDoParallel(cl)

res.df <- foreach(i = 1:length(genes),
                  .packages = c('limma'), .combine=rbind) %dopar% {
          gene = genes[i]
          
          cell_lines_wt = wes[wes[gene]==0, "Cell_line"]
          cell_lines_mut = wes[wes[gene]>0,"Cell_line"]
          
          ms_wt = meta[meta$Cell_line %in% cell_lines_wt, "Automatic_MS_filename"]
          ms_mut = meta[meta$Cell_line %in% cell_lines_mut, "Automatic_MS_filename"]
          
          protein_wt = protein[,colnames(protein) %in% ms_wt]
          protein_mut = protein[,colnames(protein) %in% ms_mut]
          
          protein_tmp = data.frame(protein[rownames(protein)==gene,], check.names = F)
          mutant = as.numeric(colnames(protein) %in% ms_mut)
          
          design <- model.matrix( ~ 1 + mutant)
          
          fit <- lmFit(protein_tmp, design)
          fit <- eBayes(fit)
          protein_res = topTable(fit, coef = ncol(design), number = nrow(protein_tmp))
          rownames(protein_res) = c(gene)
          return(protein_res)
        }
#stop cluster
stopCluster(cl)

write.table(res.df, "../../result_files/de/protein_signle_rep.csv", sep = ",", quote = F)

