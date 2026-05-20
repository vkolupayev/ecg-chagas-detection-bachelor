# help(shapiro.test)
# help(bartlett.test)
# help(t.test)
# help(wilcox.test)
# help(cor.test)

recalls <- read.csv("./evaluations/experiments_recall.csv", header = TRUE, sep = ",")
f2s <- read.csv("./evaluations/experiments_f2.csv", header = TRUE, sep = ",")
#  [1] "hubert_fine_tune_samiptb_f2_10"   "hubert_train"
#  [3] "hubert_fine_tune"                 "founder_fine_tune_samiptb_f2"
#  [5] "founder_fine_tune"                "founder_fine_tune_samiptb_f2_10"
#  [7] "founder_train"                    "hubert_fine_tune_samiptb_recall"
#  [9] "founder_linear_probing"           "hubert_linear_probing"
# [11] "founder_fine_tune_samiptb_recall"

summary(recalls)
summary(f2s)

cat("Foundational Model Comparison Experiment \n\n")

cat("Shapiro-Wilk Normality Test\n\n")
cat("The null-hypothesis of this test is that the population is normally distributed.\n")
cat("If the p value is less than the chosen alpha level, then the null hypothesis is rejected and there is evidence that the data tested are not normally distributed.\n")
cat("If the sample size is sufficiently large this test may detect even trivial departures from the null hypothesis\n")
cat("Additional investigation of the effect size is typically advisable, e.g., a Q–Q plot in this case.\n")

cat("Recall Normality tests \n\n")
shapiro.test(recalls$founder_fine_tune)
shapiro.test(recalls$hubert_fine_tune)
shapiro.test(recalls$founder_fine_tune_samiptb_recall)
shapiro.test(recalls$hubert_fine_tune_samiptb_recall)
shapiro.test(recalls$founder_train)
shapiro.test(recalls$hubert_train)
shapiro.test(recalls$founder_linear_probing)
shapiro.test(recalls$hubert_linear_probing)

cat("F2-Score Normality tests \n\n")
shapiro.test(f2s$founder_fine_tune)
shapiro.test(f2s$hubert_fine_tune)
shapiro.test(f2s$founder_fine_tune_samiptb_recall)
shapiro.test(f2s$hubert_fine_tune_samiptb_recall)
shapiro.test(f2s$founder_train)
shapiro.test(f2s$hubert_train)
shapiro.test(f2s$founder_linear_probing)
shapiro.test(f2s$hubert_linear_probing)

cat("Fine Tune Test\n\n")

cat("Wilcoxon Signed Rank Test is similar to Paired sample t-test\n")
cat("Both assume dependant paired groups. But the observations are independent of one another.\n")
cat("Applies in a within-subjects study design, i.e., in a study where the same set of subjects undergo both of the conditions being compared.\n")
cat("Paired sample t-test assumes normally distributed dependant variables, \n")
cat("as well as equal variances - homoscedasticity (or homogeneity of variance).\n")
cat("Testable using F-test, Levene's test, Bartlett's test, or the Brown–Forsythe test; or assessable graphically using a Q–Q plot.\n")
cat("In Welch's unequal variances t-test this assumption is dropped, however, these tests are often referred to as unpaired or independent samples t-tests.\n")
cat("Thus, Wilcoxon Signed Rank Test is used as it does not assume a normal distribution or equal variances.\n\n")

print("Founder Confidence Interval")
result <- t.test(recalls$founder_fine_tune, conf.level = 0.99)
confidence_interval <- result$conf.int
print(confidence_interval)

print("HuBERT Confidence Interval")
result <- t.test(recalls$hubert_fine_tune)
confidence_interval <- result$conf.int
print(confidence_interval)

# Correlation test
cat("Pearson's product-moment correlation to test association or correlation between the paired samples.\n\n")
cor.test(x = recalls$founder_fine_tune, y = recalls$hubert_fine_tune, 
         method = c("pearson"), 
         conf.level = 0.99)

cat("Recall Variance tests \n\n")
var_test_data <- data.frame(fold = rep(c(1:10), 2),
                   condition = rep(c("founder", "hubert"), each = 10),
                   score = c(recalls$founder_fine_tune, recalls$hubert_fine_tune))

bartlett.test(var_test_data$score ~ var_test_data$condition)
cat("Bartlett's test is used to test the null hypothesis that all k population variances are equal against the alternative that at least two are different.\n")
# finetuning setting passes both normality and variances thus in theory a paired t test could be used as all of the assumptions are 

cat("Paired sample t-test\n\n")

t.test(x = recalls$founder_fine_tune,
       y = recalls$hubert_fine_tune,
       alternative = "two.sided",
       mu = 0, 
       paired = TRUE,   
       var.equal = TRUE,
       conf.level = 0.99)

t.test(x = recalls$founder_fine_tune,
       y = recalls$hubert_fine_tune,
       alternative = "greater",
       mu = 0, 
       paired = TRUE,   
       var.equal = TRUE,
       conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")

wilcox.test(recalls$founder_fine_tune, recalls$hubert_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune, recalls$hubert_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$hubert_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$hubert_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Sami-PTB Fine Tune Test\n\n")

wilcox.test(recalls$founder_fine_tune_samiptb_recall,
            recalls$hubert_fine_tune_samiptb_recall,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_samiptb_recall,
            recalls$hubert_fine_tune_samiptb_recall,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_samiptb_recall,
            f2s$hubert_fine_tune_samiptb_recall,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_samiptb_recall,
            f2s$hubert_fine_tune_samiptb_recall,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Scratch Train Test\n\n")

wilcox.test(recalls$founder_train, recalls$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_train, recalls$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_train, f2s$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_train, f2s$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Linear Probe Test\n\n")

wilcox.test(recalls$founder_linear_probing, recalls$hubert_linear_probing,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_linear_probing, recalls$hubert_linear_probing,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_linear_probing, f2s$hubert_linear_probing,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_linear_probing, f2s$hubert_linear_probing,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")

cat("Founder Pre-Train Test\n\n")

wilcox.test(recalls$founder_fine_tune, recalls$founder_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune, recalls$founder_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$founder_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$founder_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("HuBERT Pre-Train Test\n\n")

wilcox.test(recalls$hubert_fine_tune, recalls$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$hubert_fine_tune, recalls$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$hubert_fine_tune, f2s$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$hubert_fine_tune, f2s$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")


cat("Foundational Model Modification Experiment \n\n")

cat("Batch Normalization\n")

wilcox.test(recalls$founder_fine_tune_bn, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_bn, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_bn, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_bn, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Dropout 25%\n")

wilcox.test(recalls$founder_fine_tune_do_25, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_do_25, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_25, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_25, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Dropout 50%\n")

wilcox.test(recalls$founder_fine_tune_do_50, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_do_50, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_50, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_50, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("Demographic Encoder\n")

wilcox.test(recalls$founder_fine_tune_de, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_de, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_de, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_de, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Discriminative LR\n")

wilcox.test(recalls$founder_fine_tune_dlr, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_dlr, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_dlr, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_dlr, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")

cat("Data Augmentation Experiment \n\n")

cat("Temporal Masking\n")

wilcox.test(recalls$founder_fine_tune_tm, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_tm, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_tm, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_tm, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Lead Masking\n")

wilcox.test(recalls$founder_fine_tune_lm, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_lm, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_lm, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_lm, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("Magnitude Scale\n")

wilcox.test(recalls$founder_fine_tune_ms, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_ms, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ms, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ms, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Gaussian Noise\n")

wilcox.test(recalls$founder_fine_tune_gn, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_gn, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_gn, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_gn, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Label Smoothing\n")

wilcox.test(recalls$founder_fine_tune_ls, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_ls, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ls, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ls, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Refined Label Smoothing\n")

wilcox.test(recalls$founder_fine_tune_rls, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_rls, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_rls, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_rls, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")
cat("\n--------------------------------------------------------------------\n")
cat("\n--------------------------------------------------------------------\n")

cat("Tests for results at 5% threshold on SaMi-Trop and upsampled PTB-XL\n\n")

recalls <- read.csv("./evaluations/experiments_subset_recalls.csv", header = TRUE, sep = ",")
f2s <- read.csv("./evaluations/experiments_subset_f2s.csv", header = TRUE, sep = ",")

summary(recalls)
summary(f2s)

cat("Foundational Model Comparison Experiment \n\n")

cat("Fine Tune Test\n\n")

wilcox.test(recalls$founder_fine_tune, recalls$hubert_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune, recalls$hubert_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$hubert_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$hubert_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Scratch Train Test\n\n")

wilcox.test(recalls$founder_train, recalls$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_train, recalls$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_train, f2s$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_train, f2s$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Linear Probe Test\n\n")

wilcox.test(recalls$founder_linear_probing, recalls$hubert_linear_probing,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_linear_probing, recalls$hubert_linear_probing,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_linear_probing, f2s$hubert_linear_probing,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_linear_probing, f2s$hubert_linear_probing,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")

cat("Founder Pre-Train Test\n\n")

wilcox.test(recalls$founder_fine_tune, recalls$founder_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune, recalls$founder_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$founder_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune, f2s$founder_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("HuBERT Pre-Train Test\n\n")

wilcox.test(recalls$hubert_fine_tune, recalls$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$hubert_fine_tune, recalls$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$hubert_fine_tune, f2s$hubert_train,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$hubert_fine_tune, f2s$hubert_train,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")


cat("Foundational Model Modification Experiment \n\n")

cat("Batch Normalization\n")

wilcox.test(recalls$founder_fine_tune_bn, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_bn, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_bn, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_bn, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Dropout 25%\n")

wilcox.test(recalls$founder_fine_tune_do_25, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_do_25, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_25, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_25, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Dropout 50%\n")

wilcox.test(recalls$founder_fine_tune_do_50, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_do_50, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_50, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_do_50, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("Demographic Encoder\n")

wilcox.test(recalls$founder_fine_tune_de, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_de, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_de, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_de, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Discriminative LR\n")

wilcox.test(recalls$founder_fine_tune_dlr, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_dlr, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_dlr, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_dlr, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("\n--------------------------------------------------------------------\n")

cat("Data Augmentation Experiment \n\n")

cat("Temporal Masking\n")

wilcox.test(recalls$founder_fine_tune_tm, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_tm, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_tm, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_tm, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Lead Masking\n")

wilcox.test(recalls$founder_fine_tune_lm, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_lm, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_lm, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_lm, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)


cat("Magnitude Scale\n")

wilcox.test(recalls$founder_fine_tune_ms, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_ms, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ms, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ms, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Gaussian Noise\n")

wilcox.test(recalls$founder_fine_tune_gn, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_gn, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_gn, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_gn, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Label Smoothing\n")

wilcox.test(recalls$founder_fine_tune_ls, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_ls, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ls, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_ls, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

cat("Refined Label Smoothing\n")

wilcox.test(recalls$founder_fine_tune_rls, recalls$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(recalls$founder_fine_tune_rls, recalls$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_rls, f2s$founder_fine_tune,
            alternative = "two.sided", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)

wilcox.test(f2s$founder_fine_tune_rls, f2s$founder_fine_tune,
            alternative = "greater", paired = TRUE,
            exact = FALSE, correct = TRUE,
            conf.int = TRUE, conf.level = 0.99)