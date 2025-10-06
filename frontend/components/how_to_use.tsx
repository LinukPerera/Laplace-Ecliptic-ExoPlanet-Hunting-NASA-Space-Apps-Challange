

export default function HowToUse() {
    const steps = [
        "Download the template and fill in your details.",
        "Upload the completed CSV file to the app.",
        "Click Proceed to start the process."
    ];


    return (
        <div className="h-screen w-full bg-black flex items-center justify-center relative overflow-hidden gradient-bg-1" id="how-to-use">
            {/*<DotLottieReact src="/anims/Space Areal.lottie" loop autoplay className="h-full w-full object-cover opacity-20" />*/}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4">
                
                <h2 className="text-2xl font-semibold mb-8 text-center text-white">How to Use the App</h2>
                <div className="space-y-6">
                    {steps.map((step, index) => (
                    <div key={index} className="flex items-center">
                        <div className="flex-shrink-0 w-10 h-10 rounded-full text-white flex items-center justify-center font-bold border border-white ">
                        {index + 1}
                        </div>
                        <p className="ml-4 text-left text-white">{step}</p>
                    </div>
                    ))}
                </div>
            </div>
        </div>
    )
};