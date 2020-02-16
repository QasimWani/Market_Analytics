const request = require("request"),
    express = require("express"),
    spawn = require("child_process").spawn,
    path = require("path"),
    router = express();
//template
// http://localhost:2718/search?query=long%20distance%20carpooling&mode=app&region=California,Virginia&age=all&price=5&year=2021

/**
 * Change this to post route later.
 */
router.get("/", (req, res)=>{
    console.log("In here...");
    return res.render("index");
});
router.post("/", (req, res)=>{
    var query = req.body.query;
    console.log(query);
    var search_data = {};

    var link = link = "https://www.google.com/search?q=" + query.toLowerCase();
    request(link, (err, body, response)=>{
        if(err || body.statusCode != 200)
        {
            throw new Error(err);
        }
        search_data.title_page = response.split("</title>")[0].split("<title>")[1];

        const process = spawn('python', [
            "-u",
            path.join(__dirname,"/main.py"),
            query
         ]);
         var some_data = "";
         process.stderr.on('data', async(data) => {
            console.log(`error: ${data}`);
            // var error = {"code": 403, "status": "File Not read properly"};
            // return res.send(error);
         });
         
         process.stderr.on('close', async() => {
            console.log("Closed");
            // var error = {"code": 402, "status": "File close read buffer type error."};
            // return res.send(error);
         });
         var some_data = "";
         process.stdout.on('data',async(data)=>{
            console.log(`data: ${data}`);
            some_data += data; 
            return res.render("results", {data : some_data, topic : query});
         });
    });
});

module.exports = router;