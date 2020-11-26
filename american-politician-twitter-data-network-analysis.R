install.packages("base64enc")
install.packages("httpuv")
install.packages("rtweet")
install.packages("tm")
install.packages("SnowballC")
install.packages("ggraph")

library("rtweet")
library("tm")
library("purrr")


key <- "G8CgmZyl8zuTxO3aBfo0WvT0w"
secret <- "XwxitW8acYI6lpN2bJnPhQ0Hy6boLKwHsa7lHArHMEkIzna6Lz"

create_token(
  app = "diviyat_test",
  consumer_key = key,
  consumer_secret = secret)

##Friends of Trump
friends <- get_friends("realDonaldTrump")
friends_id <- lookup_users(friends$user_id)
name <- friends_id$name
handle <- friends_id$screen_name
friend_followers <- friends_id$followers_count
friend_ids <- friends_id$user_id
djt_friends <- data.frame(name,handle,friend_followers,friend_ids)
only_people <- djt_friends[-c(4,6,10,15,16,20,25,28,36:41),]
new_djt_friends <- only_people[order(-only_people$friend_followers),] #sort by followers descending
most_followers <- new_djt_friends[1:20,] #get 20 friends with most followers
most_followers


##Followers of Trump
followers <- get_followers("realDonaldTrump", n=75000,retryonratelimit = TRUE)
followers_id <- lookup_users(followers$user_id)
f_name <- followers_id$name
f_handle <- followers_id$screen_name
f_followers <- followers_id$followers_count
f_ids <- followers_id$user_id
djt_followers <- data.frame(f_name,f_handle,f_followers,f_ids)
new_djt_followers <- djt_followers[order(-f_followers),] #sort by followers descending
f_most <- new_djt_followers[1:20,] #get 20 friends with most followers
f_most

#Bypassing Trump

friend_usrs <- as.vector(most_followers[['handle']]) #creating vector of screen names for 20 friends of Trump
follower_usrs <- as.vector(f_most[['f_handle']])#creating vector of screen names for 20 followers of Trump
usrs<-c(friend_usrs,follower_usrs) #all friends and followers

friend_ids <- as.vector(most_followers[['friend_ids']]) #creating vector of user ids for 20 friends of Trump
follower_ids <- as.vector(f_most[['f_ids']])#creating vector of user ids for 20 followers of Trump
usrs_ids<-c(friend_ids,follower_ids) #all friends and followers user ids

bind_usrs <-data.frame(usrs,usrs_ids)
names(bind_usrs)=c("usrs","user_id") #data frame with all 40 usernames and matching user ids

check_friends1 <- map(usrs[1:15],get_friends) #get ALL friends for first 15 usrs
names(check_friends1) <- usrs[1:15]
df1 <- do.call(rbind, lapply(check_friends1, as.data.frame)) #convert to data frame
filter_df1 <- df1[df1$user_id %in% usrs_ids, ] #filter out users friends who are not in list of friends and followers of Trump

check_friends2 <- map(usrs[16:30],get_friends) #get ALL friends for next 15 usrs
names(check_friends2) <- usrs[16:30]
df2 <- do.call(rbind, lapply(check_friends2, as.data.frame))
filter_df2 <- df2[df2$user_id %in% usrs_ids, ]

check_friends3 <- map(usrs[31:40],get_friends) #get ALL friends for last 10 usrs
names(check_friends3) <- usrs[31:40]
df3 <- do.call(rbind, lapply(check_friends1, as.data.frame)) #convert to data frame
filter_df3 <- df3[df3$user_id %in% usrs_ids, ]

all_dfs <- rbind(filter_df1,filter_df2,filter_df3)

all_dfs1 <- merge(all_dfs, bind_usrs, by="user_id") #match friend ids to screen name in new data frame
all_dfs1 <- all_dfs1[-1]

nf <- c(all_dfs$user,usrs)
nf1 <- nf[!(duplicated(nf)|duplicated(nf, fromLast=TRUE))] #getting remaining users with no edges
v <- rep(NA, length(nf1))
no_friends <- data.frame(nf1,v)
names(no_friends) <- c("user","usrs")
final_df <- rbind(all_dfs1,no_friends)


g <- graph_from_data_frame(final_df)
g1 <- as.undirected(g, mode= "mutual",edge.attr.comb="ignore") #Take only mutual edges (both follow each other)
set.seed(23)
par(mar = c(0,0,0,0))
V(g1)$label.cex = 0.9
V(g1)$label.color = "blue"
V(g1)$label.font = 1
V(g1)$label.dist = 0.8
plot(g1, layout = layout.fruchterman.reingold, vertex.size = 3,edge.curved=0)



##### Then determine if any of the friends and followers should be friends, 
##### based on their background, and add those edges to the graph.
adj_m<- as_adjacency_matrix(g1)
adj_m[1:18,1:18] <- 1
adj_m[19,23] <- 1 #create edge between Glenn Kessler and Lalah Hathaway
adj_m[19,7] <- 1
adj_m[20:21,1:18] <- 1
adj_m <- adj_m[1:40,1:40]

g4 <- graph.adjacency(adj_m, mode="undirected")
g4<- igraph::simplify(g4, remove.loops=TRUE) #remove edge loops (nodes that link to themselves)

set.seed(30)
#formatting
par(mar = c(0,0,0,0))
V(g4)$label.cex = 0.9
V(g4)$label.color = "blue"
V(g4)$label.font = 1
V(g4)$label.dist = 0.8
#plot graph
plot(g4, layout=layout.fruchterman.reingold, vertex.size=3)

#Graph Statisitcs
#Diameter
diameter(g4)

#Density
graph.density(g4)

#Neighbourhood Overlap
#get the neighbourhoods
gn <- neighborhood(g4, order = 1) #order= 1 for direct neighbours
names(gn) <-colnames(adj_m)
#get pair of nodes that are at the end of each edge
g.ends <- ends(g4, E(g4)) 
n <- nrow(g.ends) #no.of edges

no <- rep(0,n)
#iterate over all edges
for (a in 1:n){
  x <- g.ends[a,1] #access connected nodes
  y <- g.ends[a,2] # take first and second element
  
  i <- length(intersect(gn[[x]],gn[[y]])) - 2 #neighbourhood overlap function, -2 bc x is neighbour of y and vice versa
  u <- length(union(gn[[x]],gn[[y]])) - 2
  no[a] <- i/u #neighbourhood overlap
}


#Graph Homophily

g5 <- set.vertex.attribute(g4, "label",index=c(1:18,20:21), value="S") #labelling Trump Supporters
g5 <- set.vertex.attribute(g5, "label",index=c(19,23), value="NS") #labelling Non- Supporters

##Homophily
#measure homophily by looking at cross type edges

class <- c(rep("S",18),"NS",rep("S",2),"NA","NS",rep("NA",17)) #labels "S" for supporters,"NS" for non-supporters and "NA" for neutral/Na
adj_mat <-get.adjacency(g5)
adj_mat
x_pos <- which(class == "S")   #get row positions of X nodes
y_pos <- which(class == "NS")
data_cross_edge_count <- sum(adj_mat[x_pos,y_pos])
data_cross_edge_count

reps<-1000
cross_edge_count <- rep(0,reps) #vector filled with 1000 zeroes

#randomised labelling steps
for(a in 1:reps){
  #randomly assign labels 
  permuted_class <- sample(class)
  
  #count cross edges
  x_pos <- which(permuted_class == "S")   #get row positions of X nodes
  y_pos <- which(permuted_class == "NS")
  
  #exisiting connections are 1 and non- existing are zero, use sum
  cross_edge_count[a] <- sum(adj_mat[x_pos,y_pos])
}

#visualise distribution
par(mar = c(2,2,3,2))
hist(cross_edge_count)
#most ofdistribution between 8 and 10

p_val <- mean(cross_edge_count < data_cross_edge_count)
p_val

#Structural Balance
M <- -as_adjacency_matrix(g5) #set all edges as negative
M <- M[c(1:21,23),c(1:21,23)] 
M[c(1:18,20:21),c(1:18,20:21)] <- 1 #positive edges between nodes with positive association with Trump
M[c(19,22),c(19,22)] <- 1 #positive edges between nodes with negative association with Trump

D <- as.matrix(1 - M) ## create a matrix of distances
I <- as.matrix(M + 1)
plot(graph_from_adjacency_matrix(I))

image(D) ## visualise the distances
## Notice the block structure
h <- hclust(as.dist(D), method = "single")
image(D[h$order, h$order]) #distance matrix
























