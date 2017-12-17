package com.example.imageprocessing;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Random;

public class MainActivity extends AppCompatActivity implements OnTouchListener, CvCameraViewListener2 {

    // Declaring camera view and other image variables
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;
    private Scalar mBlobColorRgba;
    private Scalar mBlobColorHsv;

    // Initializing coordinates
    double x = -1;
    double y = -1;

    // Declaring textview variables for coordinates and color
    private TextView touch_coordinates;
    private TextView touch_color;

    // Declaring filter value variable and textview for filter names
    private int currentFilter;
    private TextView filterName;

    /*
    * Loads camera to the Camera Bridge View
    * */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    /*
    * Initializes variables when activity is created
    * */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        touch_coordinates = (TextView)findViewById(R.id.touch_coordinates);
        touch_color = (TextView)findViewById(R.id.touch_color);

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        filterName = (TextView)findViewById(R.id.filter_textview);
        currentFilter = 0;
    }

    /*
    * Releases camera view on pause
    * */
    @Override
    protected void onPause() {
        super.onPause();
        if(mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    /*
    * Reloads camera view on resume
    * */
    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    /*
    * Releases camera view on destroy
    * */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    /*
    * Displays the coordinates and color of the touched location when a touch event occurs
    * */
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        double yLow = (double)mOpenCvCameraView.getHeight() * 0.2401961;
        double yHigh = (double)mOpenCvCameraView.getHeight() * 0.7696078;
        double xScale = (double)cols / (double)mOpenCvCameraView.getWidth();
        double yScale = (double)rows / (yHigh - yLow);

        x = event.getX();
        y = event.getY();

        y = y - yLow;
        x = x * xScale;
        y = y * yScale;

        if((x < 0) || (y < 0) || (x > cols) || (y > rows)) {
            return false;
        }

        touch_coordinates.setText("X: " + Double.valueOf(x) + ", Y: " + Double.valueOf(y));

        Rect touchedRect = new Rect();

        touchedRect.x = (int)x;
        touchedRect.y = (int)y;
        touchedRect.width = 8;
        touchedRect.height = 8;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);
        Mat touchedRegionHsv = new Mat();

        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL, 4);
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);

        int pointCount = touchedRect.width * touchedRect.height;
        for(int i = 0; i < mBlobColorHsv.val.length; i++) {
            mBlobColorHsv.val[i] /= pointCount;
        }

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);

        touch_color.setText("Color: #" + String.format("%02X", (int)mBlobColorRgba.val[0])
                + String.format("%02X", (int)mBlobColorRgba.val[1])
                + String.format("%02X", (int)mBlobColorRgba.val[2]));

        touch_color.setTextColor(Color.rgb((int)mBlobColorRgba.val[0],
                (int)mBlobColorRgba.val[1],
                (int)mBlobColorRgba.val[2]));
        touch_coordinates.setTextColor(Color.rgb((int)mBlobColorRgba.val[0],
                (int)mBlobColorRgba.val[1],
                (int)mBlobColorRgba.val[2]));

        return false;
    }

    /*
    * Matrices are initialized after camera is started
    * */
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
    }

    /*
    * mRgba matrix is released after camera is stopped
    * */
    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    /*
    * When a frame from camera is capture, the frame is modified according
    * to the selected filter and returned
    * */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Bitmap bmp;
        Mat tmp;

        switch (currentFilter) {
            case 1:
                Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGB2GRAY, 4);
                break;

            case 2:
                bmp = null;
                tmp = new Mat (mRgba.rows(), mRgba.cols(), CvType.CV_8U, new Scalar(4));
                try {
                    Imgproc.cvtColor(mRgba, tmp, Imgproc.COLOR_RGB2RGBA, 4);
                    bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(tmp, bmp);
                } catch (CvException e){
                    Log.d("Exception",e.getMessage());
                }
                Utils.bitmapToMat(filterShadeEffect(bmp, -15000), tmp);
                Imgproc.cvtColor(tmp, mRgba, Imgproc.COLOR_RGBA2RGB);
                break;

            case 3:
                Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2BGR, 4);
                break;

            case 4:
                bmp = null;
                try {
                    bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.RGB_565);
                    Utils.matToBitmap(mRgba, bmp);
                } catch (CvException e){
                    Log.d("Exception",e.getMessage());
                }
                Utils.bitmapToMat(filterSnowEffect(bmp), mRgba);
                break;

            case 5:
                Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGB2YUV, 4);
                break;

            case 6:
                Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGB2HLS, 4);
                break;

            default:
                break;
        }

        return mRgba;
    }

    /*
    * Adds comic filter to the bitmap of mRgba
    * */
    private Bitmap filterShadeEffect(Bitmap src, int shade) {
        int width = src.getWidth();
        int height = src.getHeight();
        int[] pixels = new int[width * height];

        src.getPixels(pixels, 0, width, 0, 0, width, height);

        int index = 0;
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                index = y * width + x;
                pixels[index] &= shade;
            }
        }

        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        output.setPixels(pixels, 0, width, 0, 0, width, height);
        return output;
    }

    /*
    * Adds snow filter to the bitmap of mRgba
    * */
    private Bitmap filterSnowEffect(Bitmap src) {
        int COLOR_MAX = 0xfff;
        int width = src.getWidth();
        int height = src.getHeight();
        int[] pixels = new int[width * height];

        src.getPixels(pixels, 0, width, 0, 0, width, height);

        Random random = new Random();

        int R, G, B, index = 0, thresHold = 50;
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                index = y * width + x;

                R = Color.red(pixels[index]);
                G = Color.green(pixels[index]);
                B = Color.blue(pixels[index]);

                thresHold = random.nextInt(COLOR_MAX);
                if(R > thresHold && G > thresHold && B > thresHold) {
                    pixels[index] = Color.rgb(COLOR_MAX, COLOR_MAX, COLOR_MAX);
                }
            }
        }

        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        output.setPixels(pixels, 0, width, 0, 0, width, height);
        return output;
    }

    /*
    * Converts the hsv scalar to rgba format
    * */
    private Scalar convertScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);

        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    /*
    * Decrements the filter variable thereby setting the screen to the previous filter
    * */
    public void previousFilter(View view) {
        if(currentFilter > 0) {
            currentFilter--;
        }

        updateFilterName();
    }

    /*
    * Increments the filter variable thereby setting the screen to the next filter
    * */
    public void nextFilter(View view) {
        if(currentFilter < 6) {
            currentFilter++;
        }

        updateFilterName();
    }

    /*
    * Updates the filter name textview after filter change
    * */
    private void updateFilterName() {
        switch (currentFilter) {
            case 0:
                filterName.setText(R.string.no_filter_text);
                break;

            case 1:
                filterName.setText(R.string.filter_1_text);
                break;

            case 2:
                filterName.setText(R.string.filter_2_text);
                break;

            case 3:
                filterName.setText(R.string.filter_3_text);
                break;

            case 4:
                filterName.setText(R.string.filter_4_text);
                break;

            case 5:
                filterName.setText(R.string.filter_5_text);
                break;

            case 6:
                filterName.setText(R.string.filter_6_text);
                break;
        }
    }

}
