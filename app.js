const express = require("express"),
      bodyParser = require("body-parser"),
      moment  = require("moment"),
      mongoose = require("mongoose"),
      compression = require("compression"),
      request = require("request"),
      methodOverride    = require("method-override"),
      dotenv        = require("dotenv"),
      expressSanitizer = require('express-sanitizer'),
      cookieSession = require("cookie-session"),
      app = express();

    
dotenv.config();
app.use(compression());
app.use(expressSanitizer());
app.use(bodyParser.urlencoded({ limit: "50mb", extended: true, parameterLimit: 50000 }))
app.set("view engine", "ejs");
app.engine('ejs', require('ejs').__express);

app.use(express.static(__dirname + '/public'));

var redirectToHTTPS = require('express-http-to-https').redirectToHTTPS
app.use(redirectToHTTPS([/localhost:(\d{4})/], [/\/insecure/], 301));

app.use(methodOverride("_method"));

app.use(cookieSession({
    maxAge : 360*3600*1000,
    keys   : ['vthacks2020.werebuildingthenextgenmarketresearchaithatdoeswonders']
}));

var MemoryStore = require("memorystore")(require('express-session'));

app.use(require("express-session")({
   secret : "scrkeyoiash389wh31207891ios=iqasimwani&jda08124sadjas.todayis14jan2020,330am.",
   store: new MemoryStore({
    checkPeriod: 86400000 // prune expires entries every 24h
  }),
   resave : false,
   saveUninitialized : false
}));

mongoose.set('useCreateIndex', true);
mongoose.set('useFindAndModify', false);

mongoose.Promise = global.Promise;
mongoose.connect("mongodb://"+process.env.mongoDB+"/marketanalytics",{ useNewUrlParser: true , useUnifiedTopology: true});

app.use(function(req, res, next){
  res.set('Cache-Control', 'no-cache, private, no-store, must-revalidate, max-stale=0, post-check=0, pre-check=0');
  next();
});

// const indexRoutes = require("./routes/index"),
//     marketplaceRoutes = require("./routes/marketplace"),
//     endpointRoutes = require("./routes/endpoint");

// app.use("/produce", endpointRoutes);
// app.use("/marketplace", marketplaceRoutes);
// app.use("/", indexRoutes);

// app.get("/", function(req, res){
//     return res.render("partials/landing/index");
// });

app.get("*",(req, res)=>{
  return res.send("Hello World");
});

app.listen(process.env.PORT || 2718, process.env.IP,()=>{
    console.log("Market Analytics Server Connected");
});
